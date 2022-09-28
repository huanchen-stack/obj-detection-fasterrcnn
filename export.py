from fasterrcnn_ori import FasterRCNN
from dataloader import ObjDetectionDataset

import copy
import os
import time
from argparse import ArgumentParser
from scipy.stats import pearsonr
import click

import torch
import torch.nn as nn
# from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torchvision import transforms, utils
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

import matplotlib.pyplot as plt

if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available else "cpu"

    # Hyperparameters
    # num_epochs = 10
    # learning_rate = 0.00001
    # batch_size = 16
    # shuffle = True
    # pin_memory = True
    # num_workers = 1
    # loss_coeff = 0.3

    # dataloader
    obj_detection_dataset = ObjDetectionDataset(fname="input.jpg")
    # validation_loader = Dataloader(dataset=obj_detection_dataset, shuffle=shuffle, batch_size=batch_size,num_workers=num_workers, pin_memory=pin_memory)
    dataloader = DataLoader(dataset=obj_detection_dataset)
    print("Dataloader Created...")

    # model
    model = FasterRCNN()
    print("Model Loaded...")

    # save model
    # m = torch.jit.script(model)
    # torch.jit.save(m, 'fasterrcnn.pt')
    torch.save(model, "fasterrcnn.pt")
    print("Model Saved...")

    # load model
    # model_loaded = torch.jit.load('fasterrcnn.pt')
    model_loaded = torch.load("fasterrcnn.pt")
    print("Model Reloaded...")

    # validation
    images = None
    for sample in dataloader:
        images = sample["images"]
        targets = sample["targets"]
        print(model_loaded(images))
    print("Validation Done...")

    print(images.shape)
    print(type(images))

    # onnx conversion
    torch.onnx.export(model_loaded, 
                  images,
                  "fasterrcnn.onnx",
                  verbose=False,
                  input_names=["actual_input"],
                  output_names=["output"],
                  export_params=True,
                  )
    print("Onnx Exported...")

    