"""
part_pip_trial (Partition Pipeline Trial for the Faster R-CNN model)
For this trial, we use the BOTEC partition result, which splits the model into two seperate parts/segments.
    (denoted as pipeline_0 and pipeline_1)
This file contains segment 0 of the model.

"""

import torch
from torch.profiler import profile, record_function, ProfilerActivity
import torchvision

import torch.nn.functional as F

from torchvision.ops import misc as misc_nn_ops
from torchvision.ops import boxes as box_ops
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool, FeaturePyramidNetwork
from torchvision.ops.poolers import MultiScaleRoIAlign
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.detection.backbone_utils import _validate_trainable_layers
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.rpn import RPNHead, concat_box_prediction_layers
from torchvision.models.detection.faster_rcnn import TwoMLPHead, FastRCNNPredictor

from collections import OrderedDict
from typing import Tuple, List, Dict, Optional, Union
from torch import nn, Tensor
import numpy as np
import csv
from sigfig import round

from timer import Clock
from memorizer import MemRec
from utils import _default_anchorgen, permute_and_flatten, _tensor_size, _size_helper


class FasterRCNN(torch.nn.Module):
    def __init__(self, partitioned=False):
        """
        Only for pretrained FasterRCNN (ResNet50 as backbone).
        In constructor, we load the model and other objects for data profiling
        and model partitions.
        Args: 
            partitioned (Optional Bool) -> whether to use the partitioned model
        Vars: 
            model: pretrained fasterrcnn model (backbone=resnet50_fpn) (eval mode)
            timer: tic toc
            memorizer: ??? try convert to tensorrt
            args: store intermediate tensors
            logger: store writable outputs (convert to csv) (return in forward() )
        """
        super(FasterRCNN, self).__init__()
        # load model (pretrained fasterrcnn)
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).eval()
        self.partitioned = partitioned

        # for profiling data
        self.timer = Clock()

        # for storing intermediate tensors and final outputs
        self.logger = {
            "layer_vertices": [],
            "dependencies": [],
        }

        # for jit (conv, ReLU should be initalized at __init__)
        self.conv = list(self.model.rpn.head.conv.modules())[2]
        self.ReLU = list(self.model.rpn.head.conv.modules())[3]

    def transform(self, images):
        # load module
        transform = self.model.transform

        # forward
        images = [images[0]]
        images, _ = transform(images)

        return images

    def backbone(self, img_tensors):
        # load module
        backbone = self.model.backbone.body
        
        # forward
        x = []
        tmp_x = img_tensors
        for name, layer in backbone.named_children():
            tmp_x = layer(tmp_x)
            if name[0:5] == "layer":
                x.append(tmp_x.clone())
        
        return x

    def get_result_from_inner_blocks(self, x: Tensor, idx: int) -> Tensor:
        fpn = self.model.backbone.fpn
        num_blocks = len(fpn.inner_blocks)
        if idx < 0:
            idx += num_blocks
        out = x
        for i, module in enumerate(fpn.inner_blocks):
            if i == idx:
                out = module(x)
        return out

    def get_result_from_layer_blocks(self, x: Tensor, idx: int) -> Tensor:
        fpn = self.model.backbone.fpn
        num_blocks = len(fpn.layer_blocks)
        if idx < 0:
            idx += num_blocks
        out = x
        for i, module in enumerate(fpn.layer_blocks):
            if i == idx:
                out = module(x)
        return out

    def fpn(self, x: List[Tensor]) -> List[Tensor]:
        # load module
        fpn = self.model.backbone.fpn

        # forward
        features_ = []

        last_inner = self.get_result_from_inner_blocks(x[-1], -1)
        features_.append(self.get_result_from_layer_blocks(last_inner, -1))

        for idx in range(len(x) - 2, -1, -1):
            # inner_{idx}
            inner_lateral = self.get_result_from_inner_blocks(x[idx], idx)
            feat_shape = inner_lateral.shape[-2:]
            # interpolate
            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")
            # addition
            last_inner = inner_lateral + inner_top_down
            # layer_{idx}
            features_.insert(0, self.get_result_from_layer_blocks(last_inner, idx))

        # extra layer
        if fpn.extra_blocks is not None:
            features_.append(F.max_pool2d(features_[-1], 1, 2, 0))

        return features_

    def rpn_parallel(self, rpn_head: torchvision.models.detection.rpn.RPNHead, feature: Tensor, i: int):
        # getting batch (unnecessary)
        # FIXME: delete this segment
        # image_range = torch.arange(1, device="cuda:0" if torch.cuda.is_available() else "cpu")
        # batch_idx = image_range[:, None]
        batch_idx = torch.tensor([[0]])

        # rpn_head
        # FIXME: clean the following segment
        # t = rpn_head.conv(feature)
        t = self.conv(feature)
        t = self.ReLU(t)

        logits = [rpn_head.cls_logits(t)]
        bbox_reg = [rpn_head.bbox_pred(t)]

        # anchor generator prep
        num_anchors_per_level_shape_tensor = [o[0].shape for o in logits]
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensor]

        # anchor generate
        grid_size = feature.shape[-2:]
        # image_size = self.images.tensors.shape[-2:]
        image_size = [800, 800]
        dtype, device = feature.dtype, feature.device
        stride = [
            torch.empty((), dtype=torch.int64, device=device).fill_(image_size[0] // grid_size[0]),
            torch.empty((), dtype=torch.int64, device=device).fill_(image_size[1] // grid_size[1]),
        ]
        anchor_generator = _default_anchorgen(i)
        anchor_generator.set_cell_anchors(dtype, device)
        anchors_ = anchor_generator.grid_anchors([grid_size], [stride])[0]

        # obj and pred bbox processing
        box_cls_per_level = logits[0]
        box_regression_per_level = bbox_reg[0]
        N, AxC, H, W = box_cls_per_level.shape
        Ax4 = box_regression_per_level.shape[1]
        A = Ax4 // 4
        C = AxC // A
        box_cls_per_level = permute_and_flatten(box_cls_per_level, N, A, C, H, W)
        box_regression_per_level = permute_and_flatten(box_regression_per_level, N, A, 4, H, W)
        box_cls_ = box_cls_per_level.flatten(0, -2)
        box_regression_ = box_regression_per_level.reshape(-1, 4)

        # get proposals
        proposal_ = self.model.rpn.box_coder.decode(box_regression_.detach(), [anchors_])
        proposal_ = proposal_.view(1, -1, 4)

        # get top n idx for each level
        num_images = 1
        device = proposal_.device
        box_cls_ = box_cls_.detach()
        box_cls_ = box_cls_.reshape(1, -1)
        level_ = [torch.full((num_anchors_per_level[0], ), i, dtype=torch.int64, device=device)]
        level_ = torch.cat(level_, 0)
        level_ = level_.reshape(1, -1).expand_as(box_cls_)
        top_n_idx_ = self.model.rpn._get_top_n_idx(box_cls_, [num_anchors_per_level[0]])

        return (
            torch.sigmoid(box_cls_[batch_idx, top_n_idx_]), 
            level_[batch_idx, top_n_idx_], 
            proposal_[batch_idx, top_n_idx_]
        )
    
    def rpn(self, features_):
        # load module
        rpn_head = self.model.rpn.head

        # foward

        # FIXME: delete the following segment
        # load and declare params
        # for rpn_head
        # convs = []
        # convs.append(rpn_head.conv)
        # device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # convs.append(torch.nn.ReLU().to(device))
        # conv = torch.nn.Sequential(*convs)

        # for anchor generator
        # get args required for selecting top 1000 for dall 5 feeatures
        objectness_prob = []
        proposals = []
        levels = []
        # getting batch (unnecessary)
        # FIXME: delete the following segment
        # image_range = torch.arange(1, device="cuda:0" if torch.cuda.is_available() else "cpu")
        # batch_idx = image_range[:, None]
        batch_idx = torch.tensor([[0]])

        # rpn_parallel
        i = 0
        for feature in features_:
            # call helper for each parallel structure
            box_cls_, level_, proposal_ = self.rpn_parallel(rpn_head, feature, i)

            # append into lists
            objectness_prob.append(box_cls_)
            levels.append(level_)
            proposals.append(proposal_)

            i += 1

        # rpn_merger
        proposals = torch.cat(proposals, dim=1)
        objectness_prob = torch.cat(objectness_prob, dim=1)
        levels = torch.cat(levels, dim=1)

        final_boxes = []
        final_scores = []
        for boxes, scores, lvl, img_shape in zip(proposals, objectness_prob, levels, [(800, 800)]):  # run only once since batch=1
            boxes = box_ops.clip_boxes_to_image(boxes, img_shape)
            keep = box_ops.remove_small_boxes(boxes, self.model.rpn.min_size)
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]
            keep = torch.where(scores >= self.model.rpn.score_thresh)[0]
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]
            keep = box_ops.batched_nms(boxes, scores, lvl, self.model.rpn.nms_thresh)
            keep = keep[: self.model.rpn.post_nms_top_n()]
            boxes, scores = boxes[keep], scores[keep]

            final_boxes.append(boxes)
            final_scores.append(scores)

        proposals, proposal_losses = final_boxes, final_scores

        return proposals, proposal_losses

    def roi_box_pool(self, features_, proposals):
        # load modules
        box_roi_pool = self.model.roi_heads.box_roi_pool
        box_head = self.model.roi_heads.box_head
        box_predictor = self.model.roi_heads.box_predictor

        # load parameters
        # image_shapes = self.images.image_sizes
        features = OrderedDict([(k, v) for k, v in zip(['0', '1', '2', '3', 'pool'], features_)])
        box_features = box_roi_pool(features, proposals, [(800, 800)])

        return box_features

    def roi_box_head(self, box_features):
        # load module
        box_head = self.model.roi_heads.box_head
        
        # load variables
        x = box_features.clone()
        # flatten should take VERY small amount of time and memory
        x = x.flatten(start_dim=1)

        # fc6
        x = F.relu(box_head.fc6(x))
        # fc7
        x = F.relu(box_head.fc7(x))
        
        # update parameters
        box_features_ = x

        return box_features_

    def roi_box_predictor(self, box_features_):
        # load module
        box_predictor = self.model.roi_heads.box_predictor

        # load parameters
        x = box_features_.clone()
        # flatten should take VERY small amount of time and memory
        x = x.flatten(start_dim=1)
        
        # cls_score
        class_logits = box_predictor.cls_score(x)

        # bbox_pred_roi_
        box_regression = box_predictor.bbox_pred(x)

        return class_logits, box_regression

    def postprocess_detection(self, class_logits, box_regression, proposals):
        # load module
        postprocess_detections = self.model.roi_heads.postprocess_detections

        # forward
        # boxes, scores, labels = postprocess_detections(
        #                             class_logits, box_regression, 
        #                             proposals, self.images.image_sizes
        #                         )
        boxes, scores, labels = postprocess_detections(
                                    class_logits, box_regression, 
                                    proposals, [(800, 800)]
                                )
        result = []
        losses = {}
        num_images = len(boxes)
        for i in range(num_images):
            result.append(
                {
                    "boxes": boxes[i],
                    "labels": labels[i],
                    "scores": scores[i],
                }
            )
        detections, detector_losses = result, losses

        return detections, detector_losses

    def roi(self, features_, proposals):
        box_features = self.roi_box_pool(features_, proposals)
        box_features_ = self.roi_box_head(box_features)
        class_logits, box_regression = self.roi_box_predictor(box_features_)
        detections, detector_losses = self.postprocess_detection(class_logits, box_regression, proposals)
        return detections, detector_losses

    def resize(self, detections):
        # load function
        def resize_boxes(boxes, original_size, new_size):
            ratios = [
                torch.tensor(s, dtype=torch.float32, device=boxes.device)
                / torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
                for s, s_orig in zip(new_size, original_size)
            ]
            ratio_height, ratio_width = ratios
            xmin, ymin, xmax, ymax = boxes.unbind(1)

            xmin = xmin * ratio_width
            xmax = xmax * ratio_width
            ymin = ymin * ratio_height
            ymax = ymax * ratio_height
            return torch.stack((xmin, ymin, xmax, ymax), dim=1)

        # forward
        result = detections
        image_shapes = [(800, 800)]
        # original_image_sizes = self.args["original_image_sizes"]
        original_image_sizes = [(224, 224)]
        for i, (pred, im_s, o_im_s) in enumerate(zip(result, image_shapes, original_image_sizes)):
            boxes = pred["boxes"]
            boxes = resize_boxes(boxes, im_s, o_im_s)
            result[i]["boxes"] = boxes

        return result

    def simulation(self, images):
        """Use the self-partitioned implementation to infer."""

        images = self.transform(images)
        x = self.backbone(images.tensors)
        features_ = self.fpn(x)
        features = OrderedDict([(k, v) for k, v in zip(['0', '1', '2', '3', 'pool'], features_)])
        proposals, proposal_losses = self.rpn(features_)
        detections, detector_losses = self.roi(features_, proposals)
        detections = self.resize(detections)

        # images, _ = self.model.transform(images)
        # features = self.model.backbone(images.tensors)
        # features_ = list(features.values())
        # proposals, proposal_losses = self.model.rpn(images, features, None)
        # detections, detector_losses = self.model.roi_heads(features, proposals, [(800, 800)], None)

        return detections

    def original(self, images):
        """Use the original implementation to infer."""

        images = [images[0]]
        detections = self.model(images)
        return detections

    def forward(self, images):
        """
        Depend on the mode:
            if self.partitioned == True:
                use self-partitioned version
            else:
                use the original implementation
        """

        # TODO: Comment this segment
        print(self.simulation(images))
        return self.original(images)

        if self.partitioned:
            return self.simulation(images), self.logger
        else:
            return self.original(images), self.logger