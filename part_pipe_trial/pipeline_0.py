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
import time
import os

from utils import _default_anchorgen, permute_and_flatten, _tensor_size, _size_helper


class Pipeline0(torch.nn.Module):
    def __init__(self):
        """
        Only for pretrained FasterRCNN (ResNet50 as backbone).
        In constructor, we load the model and other objects for data profiling
        and model partitions.
        Args: 
            partitioned (Optional Bool) -> whether to use the partitioned model
        Vars: 
            model: pretrained fasterrcnn model (backbone=resnet50_fpn) (eval mode)
            timestamps: record important timestamps when intermediate results are computed
        """
        super(Pipeline0, self).__init__()
        # load model (pretrained fasterrcnn)
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="FasterRCNN_ResNet50_FPN_Weights.COCO_V1").eval()
        self.images = None
        self.path = os.getcwd()
        self.path = os.path.join(self.path, "part_pipe_trial/intermediate/")

        # for recording important timestamps when intermediate results are computed
        self.start_elapsing = None
        self.mode = None
        self.timestamps = {}
        self.intermediate = {}  # load from prev pipe

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

    def fpn_partial(self, x: List[Tensor]) -> List[Tensor]:
        # forward
        features_ = []

        last_inner = self.get_result_from_inner_blocks(x[-1], -1)
        if self.mode == "time":
            print(f"recording timestamp for: inner_{3}")
            self.timestamps[f"inner_{3}"] = time.time() - self.start_elapsing
        else:
            print(f"exporting intermediate tensors for: inner_{3}")
            torch.save(last_inner, os.path.join(self.path, f"inner_{3}.pt"))
        # features_.append(self.get_result_from_layer_blocks(last_inner, -1))

        for idx in range(len(x) - 2, -1, -1):
            # inner_{idx}
            inner_lateral = self.get_result_from_inner_blocks(x[idx], idx)
            feat_shape = inner_lateral.shape[-2:]
            # interpolate
            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")
            # addition
            last_inner = inner_lateral + inner_top_down
            if idx == 1 or idx == 2:
                if self.mode == "time":
                    print(f"recording timestamp for: add__{idx}")
                    self.timestamps[f"add__{idx}"] = time.time() - self.start_elapsing
                else:
                    print(f"exporting intermediate tensors for: add__{idx}")
                    torch.save(last_inner, os.path.join(self.path, f"add__{idx}.pt"))
            else:
                features_.insert(0, self.get_result_from_layer_blocks(last_inner, idx))
            # # layer_{idx}
            # features_.insert(0, self.get_result_from_layer_blocks(last_inner, idx))

        # # extra layer
        # if fpn.extra_blocks is not None:
        #     features_.append(F.max_pool2d(features_[-1], 1, 2, 0))

        return features_

    def rpn_parallel(self, rpn_head: torchvision.models.detection.rpn.RPNHead, feature: Tensor, i: int):
        # getting batch
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
    
    def rpn_partial(self, features_):
        # load module
        rpn_head = self.model.rpn.head

        # forward
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

        if self.mode == "time":
            print(f"recording timestamp for: rpn_parallel_f0")
            self.timestamps[f"rpn_parallel_f0"] = time.time() - self.start_elapsing
        else:
            print(f"exporting intermediate tensors for: rpn_parallel_f0")
            rpn_parallel_f0 = OrderedDict([(k, v) for k, v in zip(['objectness_prov', 'levels', 'proposals'], [objectness_prob[0], levels[0], proposals[0]])])
            torch.save(rpn_parallel_f0, os.path.join(self.path, f"rpn_parallel_f0.pt"))

    def simulation_mode(self, mode):
        self.mode = mode
        if self.mode == "time":
            self.start_elapsing = time.time()

        images = self.images

        images = self.transform(images)
        x = self.backbone(images.tensors)
        features_ = self.fpn_partial(x)
        self.rpn_partial(features_)

    def simulation(self):
        self.simulation_mode("time")
        self.simulation_mode("export")

    def forward(self, images):
        self.images = images
        self.simulation()
        return self.timestamps