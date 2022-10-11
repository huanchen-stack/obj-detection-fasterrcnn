from gc import enable
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

from util.timer import Clock
from util.memorizer import MemRec
from util.utils import _default_anchorgen, permute_and_flatten, _tensor_size, _size_helper


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
        if torchvision.__version__ == '0.13.0':
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='FasterRCNN_ResNet50_FPN_Weights.COCO_V1').eval()
        else:
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).eval()
        self.partitioned = partitioned

        # for profiling data
        self.timer = Clock()
        self.memorizer = MemRec()
        self.mem = False  # get memory consumption
        self.warmup = 3  # num of warmup iterations

        # for storing intermediate tensors and final outputs
        self.logger = {
            "profiles": [],
            "dependencies": [],
        }

        # for jit (conv, ReLU should be initalized at __init__)
        if torchvision.__version__ == '0.13.0':
            self.conv = list(self.model.rpn.head.conv.modules())[2]
            self.ReLU = list(self.model.rpn.head.conv.modules())[3]
        else:
            self.conv = self.model.rpn.head.conv
            self.ReLU = torch.nn.ReLU()

    def _profiler_wrapper(self, func, name, ignore_payload=False):
        """
        This function is a profiling helper that does the following:
            1. warm up gpu n (warmup) times
            2. get runtime (WALL)
        Args:
            1. (layer) func (Function): 
                a. MUST NOT modify any intermediate variables
                b. MUST return the output of corresponding layer
                c. output must be a Tensor or a List of Tensors
            2. name (String): MUST be consistant with the dependencies
        """
        # warm up (default twice)
        for i in range(self.warmup):
            output = func()

        # get layer runtime (WALL)
        self.timer.tic()
        func()
        self.timer.toc()

        if self.mem: # get layer memory consumption
            with profile(
                    activities=
                    [
                        ProfilerActivity.CPU
                    ] if not torch.cuda.is_available() else
                    [
                        ProfilerActivity.CPU,
                        ProfilerActivity.CUDA
                    ],
                    profile_memory=True, record_shapes=True
                ) as prof:
                with record_function("model_inference"):
                    func()
            prof_report = str(prof.key_averages().table()).split("\n")
            mem_out = self.memorizer.get_mem(prof_report, torch.cuda.is_available())

        # output format:
        #   layername,time,mem_cpu,mem_cuda,size,macs
        #   undefined: -1
        data_payload = 0 if ignore_payload else _size_helper(output)[1]
        if not self.mem:
            self.logger["profiles"].append(f"{name},{self.timer.get_time()},-1,-1,{data_payload},-1")
        elif self.mem and torch.cuda.is_available():
            self.logger["profiles"].append(f"{name},{self.timer.get_time()},{mem_out[0]},{mem_out[1]},{data_payload},-1")
        elif self.mem and not torch.cuda.is_available():
            self.logger["profiles"].append(f"{name},{self.timer.get_time()},{mem_out},-1,{data_payload},-1")
        else:
            assert False, f"Profiler: unknown profiling configurations."
        
    def _dependency_writer(self, source, target):
        self.logger["dependencies"].append(f"{source},{target}")

    def _logger_print(self):
        print("================================================")
        print("=================== PROFILES ===================")
        print("================================================")
        for line in self.logger["profiles"]:
            print(line)
        print()
        print("================================================")
        print("================== DEPENDENCIES ================")
        print("================================================")
        for line in self.logger["dependencies"]:
            print(line)

    def _logger_write(self):
        f = open("tmp.csv", 'a')
        for line in self.logger["profiles"]:
            f.write(f"{line}\n")
        f.write(f"input,0,0,0,0.6021,0\n")
        # f.write(f"resize,0,0,0,0,0\n")
        f.write(f"output,0,0,0,0,0\n")
        f.write(f"add__2,0,0,0,2.56,0\n")
        f.write(f"add__1,0,0,0,10.24,0\n")
        f.write(f"add__0,0,0,0,40.96,0\n")
        f.close()

    def transform(self, images):
        # load module
        transform = self.model.transform

        # forward
        #   setup variables
        images = [images[0]]
        #   setup func
        def _transform():
            return transform(images, None)[0]
        self._profiler_wrapper(_transform, "transform")
        #   get layer outputs
        images_ = _transform()
        #   logger outputs
        self._dependency_writer("input", "transform")

        return images_

    def backbone(self, img_tensors):
        # load module
        backbone = self.model.backbone.body
        
        # forward
        #   setup variables
        x = []
        tmp_x = img_tensors
        last_name = "transform"
        for name, layer in backbone.named_children():
            # setup func
            def _layer():
                return layer(tmp_x)
            self._profiler_wrapper(_layer, name)
            tmp_x_ = _layer()
            # logger outputs
            self._dependency_writer(last_name, name)
            last_name = name
            # get layer outputs
            tmp_x = tmp_x_
            if name[0:5] == "layer":
                x.append(tmp_x.clone().detach())
        
        return x

    def get_result_from_inner_blocks(self, x: Tensor, idx: int) -> Tensor:
        fpn = self.model.backbone.fpn
        num_blocks = len(fpn.inner_blocks)
        if idx < 0:
            idx += num_blocks
        out = x
        for i, module in enumerate(fpn.inner_blocks):
            if i == idx:
                # forward
                #   setup func
                def _inner():
                    return module(x)
                self._profiler_wrapper(_inner, f"inner_{idx}")
                #   get layer outputs
                out = _inner()
                #   logger outputs

        return out

    def get_result_from_layer_blocks(self, x: Tensor, idx: int) -> Tensor:
        fpn = self.model.backbone.fpn
        num_blocks = len(fpn.layer_blocks)
        if idx < 0:
            idx += num_blocks
        out = x
        for i, module in enumerate(fpn.layer_blocks):
            if i == idx:
                # forward
                #   setup func
                def _layer():
                    return module(x)
                self._profiler_wrapper(_layer, f"layer_{idx}")
                #   get layer outputs
                out = _layer()
                #   logger outputs
        
        return out

    def fpn(self, x: List[Tensor]) -> List[Tensor]:
        # load module
        fpn = self.model.backbone.fpn

        # forward
        features_ = []

        last_inner = self.get_result_from_inner_blocks(x[-1], -1)
        features_.append(self.get_result_from_layer_blocks(last_inner, -1))
        self._dependency_writer("layer4", "inner_3")
        self._dependency_writer("inner_3", "layer_3")

        for idx in range(len(x) - 2, -1, -1):
            # inner_{idx}
            inner_lateral = self.get_result_from_inner_blocks(x[idx], idx)
            feat_shape = inner_lateral.shape[-2:]
            self._dependency_writer(f"layer{idx+1}", f"inner_{idx}")

            # interpolate
            #   setup func
            def _interpolate():
                return F.interpolate(last_inner, size=feat_shape, mode="nearest")
            self._profiler_wrapper(_interpolate, f"interpolate__{idx}")
            #   get layer output
            inner_top_down = _interpolate()
            #   logger output
            if idx == 2:
                self._dependency_writer("inner_3", "interpolate__2")
            else:
                self._dependency_writer(f"add__{idx+1}", f"interpolate__{idx}")

            # addition
            last_inner = inner_lateral + inner_top_down
            self._dependency_writer(f"inner_{idx}", f"add__{idx}")
            self._dependency_writer(f"interpolate__{idx}", f"add__{idx}")

            # layer_{idx}
            features_.insert(0, self.get_result_from_layer_blocks(last_inner, idx))
            self._dependency_writer(f"add__{idx}", f"layer_{idx}")

        # extra layer
        if fpn.extra_blocks is not None:
            # (ALWAYS THIS CASE)
            # forward
            #   setup func
            def _extra():
                return F.max_pool2d(features_[-1], 1, 2, 0)
            self._profiler_wrapper(_extra, "extra")
            #   get layer outputs
            features_.append(F.max_pool2d(features_[-1], 1, 2, 0))
            #   logger outputs
            self._dependency_writer("layer_3", "extra")

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
            # forward
            #   setup func
            def _rpn_parallel():
                return self.rpn_parallel(rpn_head, feature, i)
            self._profiler_wrapper(_rpn_parallel, f"rpn_parallel_f{i}")
            #   get layer outputs
            box_cls_, level_, proposal_ = self.rpn_parallel(rpn_head, feature, i)
            #   logger outputs
            if i != 4:
                self._dependency_writer(f"layer_{i}", f"rpn_parallel_f{i}")
            else:
                self._dependency_writer("extra", "rpn_parallel_f4")

            # append into lists
            objectness_prob.append(box_cls_)
            levels.append(level_)
            proposals.append(proposal_)

            # # get more details
            # self.rpn_parallel_details(rpn_head, feature, i)

            i += 1

        # rpn_merger
        #   this segment should not take significant amount of time
        proposals = torch.cat(proposals, dim=1)
        objectness_prob = torch.cat(objectness_prob, dim=1)
        levels = torch.cat(levels, dim=1)

        # forward
        #   setup func
        def _rpn_merger():
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
        self._profiler_wrapper(_rpn_merger, "rpn_merger")
        #   get layer outputs
        proposals, proposal_losses = _rpn_merger()
        #   logger outputs
        for i in range(len(features_)):
            self._dependency_writer(f"rpn_parallel_f{i}", "rpn_merger")

        return proposals, proposal_losses

    def roi_box_pool(self, features_, proposals):
        # load modules
        box_roi_pool = self.model.roi_heads.box_roi_pool
        box_head = self.model.roi_heads.box_head
        box_predictor = self.model.roi_heads.box_predictor

        # load parameters
        # image_shapes = self.images.image_sizes
        features = OrderedDict([(k, v) for k, v in zip(['0', '1', '2', '3', 'pool'], features_)])

        # forward
        #   setup func
        def _roi_box_pool():
            return box_roi_pool(features, proposals, [(800, 800)])
        self._profiler_wrapper(_roi_box_pool, "box_roi_pool")
        #   get layer outputs
        box_features = _roi_box_pool()
        #   logger outputs
        for i in range(len(features_)):
            if i != 4:
                self._dependency_writer(f"layer_{i}", "box_roi_pool")
            else:
                self._dependency_writer("extra", "box_roi_pool")
        self._dependency_writer("rpn_merger", "box_roi_pool")

        return box_features

    def roi_box_head(self, box_features):
        # load module
        box_head = self.model.roi_heads.box_head
        
        # load variables
        x = box_features.clone().detach()
        # flatten should take VERY small amount of time and memory
        x = x.flatten(start_dim=1)

        # fc6
        def _fc6():
            return F.relu(box_head.fc6(x))
        self._profiler_wrapper(_fc6, "fc6")
        self._dependency_writer("roi_box_pool", "fc6")
        x = _fc6()
        # x = F.relu(box_head.fc6(x))

        # fc7
        def _fc7():
            return F.relu(box_head.fc7(x))
        self._profiler_wrapper(_fc7, "fc7")
        self._dependency_writer("fc6", "fc7")
        x = _fc7()
        # x = F.relu(box_head.fc7(x))
        
        # update parameters
        box_features_ = x

        return box_features_

    def roi_box_predictor(self, box_features_):
        # load module
        box_predictor = self.model.roi_heads.box_predictor

        # load parameters
        x = box_features_.clone().detach()
        # flatten should take VERY small amount of time and memory
        x = x.flatten(start_dim=1)
        
        # cls_score
        def _cls_score():
            return box_predictor.cls_score(x)
        self._profiler_wrapper(_cls_score, "cls_score")
        class_logits = _cls_score()
        self._dependency_writer("fc7", "cls_score")
        # class_logits = box_predictor.cls_score(x)

        # bbox_pred_roi_
        def _bbox_pred():
            return box_predictor.bbox_pred(x)
        self._profiler_wrapper(_bbox_pred, "bbox_pred")
        box_regression = _bbox_pred()
        self._dependency_writer("fc7", "bbox_pred")
        # box_regression = box_predictor.bbox_pred(x)

        return class_logits, box_regression

    def postprocess_detections(self, class_logits, box_regression, proposals):
        # load module
        postprocess_detections = self.model.roi_heads.postprocess_detections

        # forward
        # boxes, scores, labels = postprocess_detections(
        #                             class_logits, box_regression, 
        #                             proposals, self.images.image_sizes
        #                         )
        def _postprocess_detections():
            return postprocess_detections(
                        class_logits, box_regression, 
                        proposals, [(800, 800)]
                    ) 
        self._profiler_wrapper(_postprocess_detections, "postprocess_detections", ignore_payload=True)
        boxes, scores, labels = postprocess_detections(
                                    class_logits, box_regression, 
                                    proposals, [(800, 800)]
                                )
        self._dependency_writer("cls_score", "postprocess_detections")
        self._dependency_writer("bbox_pred", "postprocess_detections")
        self._dependency_writer("rpn_merger", "postprocess_detections")

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
        detections, detector_losses = self.postprocess_detections(class_logits, box_regression, proposals)
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
        
        self._dependency_writer("postprocess_detections", "resize")
        self._dependency_writer("resize", "output")

        return result

    def simulation(self, images):
        """Use the self-partitioned implementation to infer."""

        images = self.transform(images)
        x = self.backbone(images.tensors)
        features_ = self.fpn(x)
        # features = OrderedDict([(k, v) for k, v in zip(['0', '1', '2', '3', 'pool'], features_)])
        proposals, _ = self.rpn(features_)
        detections, _ = self.roi(features_, proposals)
        detections = self.resize(detections)

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

        # # DEBUG MODE: Uncomment this segment
        print(self.simulation(images))
        # self._logger_print()
        self._logger_write()
        print(self.original(images))
        print("\t\t\t", self.timer.get_agg())


        
        self.timer.tic()
        self.original(images)
        self.timer.toc()
        print(self.timer.get_time())
        
        # start = time.time()
        # self.original(images)
        # end = time.time()
        # print(f"delta: {end - start}")
        # print(F"{start},{end}")
        return

        if self.partitioned:
            return self.simulation(images), self.logger
        else:
            return self.original(images), self.logger


from PIL import Image
from torchvision import transforms

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


if __name__ == "__main__":
    
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

    # erase tmp.csv
    f = open("tmp.csv", "w")
    f.close()
    
    for i in range(3):
        fasterrcnn(images)

    # clean up
    d = {}
    f = open("tmp.csv", "r")
    reader = csv.reader(f, delimiter=',')
    for layername, time, _, _, data, _ in reader:
        if layername not in d:
            d[layername] = {
                "time": [],
                "data": data,
            }
        d[layername]["time"].append(float(time))
    f.close()
    f = open("tmp.csv", "w")
    f.write("layer_name,time,cpu_mem,cuda_mem,size,macs\n")
    for key, val in d.items():
        val["time"] = round(sum(val["time"])/len(val["time"]), 6)
        f.write(f"{key},{val['time']},-1,-1,{val['data']},-1\n")
    f.close()
