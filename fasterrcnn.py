import torch
from torch import nn
from torch.profiler import profile, record_function, ProfilerActivity
import torchvision

from collections import OrderedDict
import numpy as np
import csv

from sigfig import round
from timer import Clock
from memorizer import MemRec

from utils import _default_anchorgen, permute_and_flatten, _tensor_size, _size_helper


class FasterRCNN(torch.nn.Module):
    def __init__(self):
        """
        Only for pretrained FasterRCNN (ResNet50 as backbone).
        In constructor, we load the model and other objects for data profiling
        and model partitions.
        """
        super(FasterRCNN, self).__init__()
        # load model (pretrained fasterrcnn)
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).eval()

        # for profiling data
        self.timer = Clock()
        self.memorizer = MemRec()

        # for storing intermediate tensors and final outputs
        self.intermediate = None
        self.out = None

    def forward(self, images):
        # print(images.shape)
        # print(type(images))
        images = [images[0]]
        # print(type(images))
        self.out = self.model(images)
        return self.out