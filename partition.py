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
import time
# from sigfig import round
import functools

from util.timer import Clock
from util.memorizer import MemRec
from util.utils import _default_anchorgen, permute_and_flatten, _tensor_size, _size_helper
from util.partition_manager import PartitionManager, partition_manager


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


        #########################################################
        ############## Needed for PartitionManager ##############
        #########################################################
        self.exec_labels = {
            "img_transform", 
            "backbone_conv1", "backbone_bn1", "backbone_relu", "backbone_maxpool", 
            "backbone_layer1", "backbone_layer2", "backbone_layer3", "backbone_layer4", 
            "fpn_inner_3", "fpn_layer_3", 
            "fpn_inner_2", "fpn_interpolate_2", "fpn_add_2", "fpn_layer_2", 
            "fpn_inner_1", "fpn_interpolate_1", "fpn_add_1", "fpn_layer_1", 
            "fpn_inner_0", "fpn_interpolate_0", "fpn_add_0", "fpn_layer_0", 
            "fpn_extra", 
            "rpn_parallel_0", "rpn_parallel_1", "rpn_parallel_2", "rpn_parallel_3", "rpn_parallel_4", 
            "rpn_merger", 
            "roi_box_pooling", 
            "roi_head_fc6", "roi_head_fc7", 
            "roi_cls_score", "roi_bbox_pred", 
            "roi_postprocess_detections",
        }
        self.args = {}
        self.filtering = False
        #########################################################

        if torchvision.__version__ == '0.13.0':
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='FasterRCNN_ResNet50_FPN_Weights.COCO_V1').eval()
        else:
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).eval()
        self.partitioned = partitioned

        if torchvision.__version__ == '0.13.0':
            self.conv = list(self.model.rpn.head.conv.modules())[2]
            self.ReLU = list(self.model.rpn.head.conv.modules())[3]
        else:
            self.conv = self.model.rpn.head.conv
            self.ReLU = torch.nn.ReLU()

    def transform(self):

        @partition_manager(self, self.filtering, "")
        def img_transform(x):
            return self.model.transform(x, None)[0]
        img_transform("images")

    def backbone(self):

        self.args['img_tensors'] = self.args["img_transform"].tensors
        prevlayer = "img_tensors"

        for name, layer in self.model.backbone.body.named_children():

            @partition_manager(self, self.filtering, suffix=f"_{name}")
            def backbone(x):
                return layer(x)
            backbone(prevlayer)

            prevlayer = f"backbone_{name}"

        self.args['x'] = [
            "backbone_layer1",
            "backbone_layer2",
            "backbone_layer3",
            "backbone_layer4",
        ]  # for convenience

    def get_result_from_inner_blocks(self, prevlayer, idx):
        num_blocks = len(self.model.backbone.fpn.inner_blocks)
        if idx < 0:
            idx += num_blocks

        for i, module in enumerate(self.model.backbone.fpn.inner_blocks):
            if i == idx:
                
                @partition_manager(self, self.filtering, suffix=f"_{idx}")
                def fpn_inner(x):
                    return module(x)
                fpn_inner(prevlayer)

        return f"fpn_inner_{idx}"

    def get_result_from_layer_blocks(self, prevlayer, idx):
        num_blocks = len(self.model.backbone.fpn.layer_blocks)
        if idx < 0:
            idx += num_blocks
        
        for i, module in enumerate(self.model.backbone.fpn.layer_blocks):
            if i == idx:
                
                @partition_manager(self, self.filtering, suffix=f"_{idx}")
                def fpn_layer(x):
                    return module(x)
                fpn_layer(prevlayer)

        return f"fpn_layer_{idx}" 

    def fpn(self):

        features_ = []

        x = self.args['x']

        last_inner = self.get_result_from_inner_blocks(x[-1], -1)
        features_.append(self.get_result_from_layer_blocks(last_inner, -1))
        
        for idx in range(len(x) - 2, -1, -1):
            # inner_{idx}
            inner_lateral = self.get_result_from_inner_blocks(x[idx], idx)
            feat_shape = self.args[inner_lateral].shape[-2:]

            # interpolate
            @partition_manager(self, self.filtering, f"_{idx}")
            def fpn_interpolate(x):
                return F.interpolate(x, size=feat_shape, mode="nearest")
            fpn_interpolate(last_inner)
            inner_top_down = f"fpn_interpolate_{idx}"

            # addition
            @partition_manager(self, self.filtering, f"_{idx}")
            def fpn_add(x, y):
                return x + y
            fpn_add(inner_lateral, inner_top_down)
            last_inner = f"fpn_add_{idx}"

            # layer_{idx}
            features_.insert(0, self.get_result_from_layer_blocks(last_inner, idx))

        # extra layer
        @partition_manager(self, self.filtering, "")
        def fpn_extra(x):
            return F.max_pool2d(x, 1, 2, 0)
        fpn_extra(features_[-1])
        
        features_.append("fpn_extra")

        self.args["features_"] = features_  # for convenience
        
    def rpn_parallel(self, feature: Tensor, i: int):
        batch_idx = torch.tensor([[0]])

        rpn_head = self.model.rpn.head

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
    
    def rpn_parallel_details(self, rpn_head, feature, i):
        # getting batch (unnecessary)
        batch_idx = torch.tensor([[0]])

        # conv
        def _conv():
            return self.ReLU(self.conv(feature))
        self._profiler_wrapper(_conv, f"rpn_parallel_f{i}:conv", ignore_payload=True)
        t = _conv()

        # cls and bbox
        def _cls():
            return [rpn_head.cls_logits(t)]
        def _bbox():
            return [rpn_head.bbox_pred(t)]
        self._profiler_wrapper(_cls, f"rpn_parallel_f{i}:cls", ignore_payload=True)
        self._profiler_wrapper(_bbox, f"rpn_parallel_f{i}:bbox", ignore_payload=True)
        logits = _cls()
        bbox_reg = _bbox()

        # anchor generator prep
        num_anchors_per_level_shape_tensor = [o[0].shape for o in logits]
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensor]

        # anchor generate
        def _anchor():
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
            return anchors_
        self._profiler_wrapper(_anchor, f"rpn_parallel_{i}:anchor")
        anchors_ = _anchor()

        # obj and pred bbox processing
        box_cls_per_level = logits[0]
        box_regression_per_level = bbox_reg[0]
        N, AxC, H, W = box_cls_per_level.shape
        Ax4 = box_regression_per_level.shape[1]
        A = Ax4 // 4
        C = AxC // A
        def _cls_permflat():
            return permute_and_flatten(box_cls_per_level, N, A, C, H, W)
        def _bbox_permflat():
            return permute_and_flatten(box_regression_per_level, N, A, 4, H, W)
        self._profiler_wrapper(_cls_permflat, f"rpn_parallel_f{i}:cls_permflat", ignore_payload=True)
        self._profiler_wrapper(_bbox_permflat, f"rpn_parallel_f{i}:bbox_permflat", ignore_payload=True)
        box_cls_per_level = _cls_permflat()
        box_regression_per_level = _bbox_permflat()

        box_cls_ = box_cls_per_level.flatten(0, -2)
        box_regression_ = box_regression_per_level.reshape(-1, 4)

        # get proposals
        def _decode():
            return self.model.rpn.box_coder.decode(box_regression_.detach(), [anchors_])
        self._profiler_wrapper(_decode, f"rpn_parallel_f{i}:decode", ignore_payload=True)
        proposal_ = _decode()
        proposal_ = proposal_.view(1, -1, 4)

        # get top n idx for each level
        num_images = 1
        device = proposal_.device
        box_cls_ = box_cls_.detach()
        box_cls_ = box_cls_.reshape(1, -1)
        level_ = [torch.full((num_anchors_per_level[0], ), i, dtype=torch.int64, device=device)]
        level_ = torch.cat(level_, 0)
        level_ = level_.reshape(1, -1).expand_as(box_cls_)
        def _top_idx():
            return self.model.rpn._get_top_n_idx(box_cls_, [num_anchors_per_level[0]])
        self._profiler_wrapper(_top_idx, f"rpn_parallel_f{i}:top_idx", ignore_payload=True)
        top_n_idx_ = _top_idx()

        def _sigmoid():
            return torch.sigmoid(box_cls_[batch_idx, top_n_idx_])
        self._profiler_wrapper(_sigmoid, f"rpn_parallel_f{i}:sigmoid", ignore_payload=True)

        return (
            torch.sigmoid(box_cls_[batch_idx, top_n_idx_]), 
            level_[batch_idx, top_n_idx_], 
            proposal_[batch_idx, top_n_idx_]
        )

    def rpn(self):
        # for anchor generator
        # get args required for selecting top 1000 for dall 5 feeatures
        objectness_prob = []
        proposals = []
        levels = []

        # rpn_parallel
        i = 0
        for feature in self.args["features_"]:
            
            @partition_manager(self, self.filtering, suffix=f"_{i}")
            def rpn_parallel(x):
                return self.rpn_parallel(x, i)  # i is for generating anchor size
            rpn_parallel(feature)

            box_cls_, level_, proposal_ = self.args[f"rpn_parallel_{i}"]

            objectness_prob.append(box_cls_)
            levels.append(level_)
            proposals.append(proposal_)

            # # get more details
            # self.rpn_parallel_details(rpn_head, feature, i)

            i += 1

        # rpn_merger
        proposals = torch.cat(proposals, dim=1)
        objectness_prob = torch.cat(objectness_prob, dim=1)
        levels = torch.cat(levels, dim=1)

        self.args["proposals"] = proposals
        self.args["objectness_prob"] = objectness_prob
        self.args["levels"] = levels

        @partition_manager(self, self.filtering, suffix=f"")
        def rpn_merger(proposals, objectness_prob, levels):
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
            return final_boxes, final_scores
        rpn_merger("proposals", "objectness_prob", "levels")

        self.args["proposals"] = self.args["rpn_merger"][0]

    def roi_box_pool(self):

        @partition_manager(self, self.filtering, suffix=f"")
        def roi_box_pooling(features_, proposals):
            features_ = [self.args[featureStr] for featureStr in features_]
            features = OrderedDict([(k, v) for k, v in zip(['0', '1', '2', '3', 'pool'], features_)])
            return self.model.roi_heads.box_roi_pool(features, proposals, [(800, 800)])
        roi_box_pooling("features_", "proposals")
        
    def roi_box_head(self):

        @partition_manager(self, self.filtering, suffix=f"")
        def roi_head_fc6(x):
            return F.relu(self.model.roi_heads.box_head.fc6(x.flatten(start_dim=1)))
        roi_head_fc6("roi_box_pooling")

        @partition_manager(self, self.filtering, suffix=f"")
        def roi_head_fc7(x):
            return F.relu(self.model.roi_heads.box_head.fc7(x))
        roi_head_fc7("roi_head_fc6")
        
    def roi_box_predictor(self):
        
        @partition_manager(self, self.filtering, suffix=f"")
        def roi_cls_score(x):
            return self.model.roi_heads.box_predictor.cls_score(x.flatten(start_dim=1))
        roi_cls_score("roi_head_fc7")
        
        @partition_manager(self, self.filtering, suffix=f"")
        def roi_bbox_pred(x):
            return self.model.roi_heads.box_predictor.bbox_pred(x.flatten(start_dim=1))
        roi_bbox_pred("roi_head_fc7")

    def postprocess_detections(self):
        # load module
        postprocess_detections = self.model.roi_heads.postprocess_detections

        # forward
        # boxes, scores, labels = postprocess_detections(
        #                             class_logits, box_regression, 
        #                             proposals, self.images.image_sizes
        #                         )
        @partition_manager(self, self.filtering, suffix=f"")
        def roi_postprocess_detections(class_logits, box_regression, proposals):
            return self.model.roi_heads.postprocess_detections(
                        class_logits, box_regression, 
                        proposals, [(800, 800)]
                    ) 
        roi_postprocess_detections("roi_cls_score", "roi_bbox_pred", "proposals")

    def roi(self):
        self.roi_box_pool()
        self.roi_box_head()
        self.roi_box_predictor()
        self.postprocess_detections()

    def simulation(self, images):
        """Use the self-partitioned implementation to infer."""

        self.args['images'] = [images[0]]
        self.transform()
        self.backbone()
        self.fpn()
        self.rpn()
        self.roi()

    def original(self, images):
        """Use the original implementation to infer."""

        images = [images[0]]
        detections = self.model(images)
        return detections

    def forward(self, images, exec_labels=None):
        """This forward function returns the execution time of a given partition."""
        self.simulation(images)     # skimming
        self.filtering = True       
        
        self.simulation(images)     # warm up (use baseline)
        self.simulation(images)     # warm up (use baseline)

        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        starter.record()
        self.simulation(images)     # baseline
        ender.record()
        torch.cuda.synchronize()
        delta_baseline = starter.elapsed_time(ender)/1000

        if exec_labels is not None:
            self.exec_labels = exec_labels        
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        starter.record()
        self.simulation(images)     # partition
        ender.record()
        torch.cuda.synchronize()
        delta_partition = starter.elapsed_time(ender)/1000

        self.exec_labels = []
        torch.cuda.synchronize()
        starter.record()
        self.simulation(images)     # find cpu overhead caused by functools
        ender.record()
        torch.cuda.synchronize()
        delta_overhead = starter.elapsed_time(ender)/1000
        
        return {
            "baseline": delta_baseline,
            "partition": delta_partition,
            "overhead": delta_overhead,
            "speed_up_rate": (delta_partition - delta_overhead) / (delta_baseline - delta_overhead)
        }

from PIL import Image
from torchvision import transforms
import json

device = "cuda:0" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    print("test input: json")
    f = open("critical_path.json", "r")
    critical_path = json.load(f)
    
    images = Image.open('input.jpg')
    images = np.array(images)
    transform = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize((224,224)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])

    images = transform(images)
    images = torch.unsqueeze(images, dim=0).to(device)
    fasterrcnn = FasterRCNN().to(device)
    
    assert critical_path[0]["type"] == "bandwidth"
    bandwidth = critical_path[0]["content"]
    critical_path.pop(0)

    for node in critical_path:
        if node["type"] == "exec":
            d = fasterrcnn(images, node["content"])
            print(d)
        elif node["type"] == "data":
            d = node["content"] / bandwidth
            print(d)
        else: 
            assert False, f"Node TYPE-{node['type']} not recognized..."
    

