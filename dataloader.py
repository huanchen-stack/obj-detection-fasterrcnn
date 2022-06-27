from __future__ import print_function, division
import os
import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import json

class ObjDetectionDataset(Dataset):
    def __init__(self, json_file=None, root_dir=None, transform=None, fname=None, iterations=None):
        """
        Input Args:
            json_file: If given, use the gt bb included in the file.
            root_dir: If given, use its images as inputs.
            transform (Optional): Optional transforms may be applied.
            fname (Optional): If given, force to use the given file under cur_dir
        """
        self.bb_gts = None
        if json_file is not None:
            file = open(json_file)
            self.bb_gts = json.load(file)
            file.close()
        
        self.root_dir = root_dir

        self.transform = transform
        if self.transform is None:
            self.transform = transforms.Compose([
                                transforms.ToPILImage(),
                                transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                            ])

        self.fname = fname
        self.iterations = iterations

    def __len__(self):
        """
        Return #iterations (default=1) if using rand gen image.
        """
        iterations = 1 if not self.iterations else self.iterations
        return iterations if self.bb_gts is None else len(self.bb_gts)

    def __getitem__(self, key):
        """
        If root_dir is not specified, use rand gen image, size = (1, 3, 224, 224).
        Otherwise, load from root_dir.
        """

        if self.fname is not None:
            images = io.imread(self.fname)
            images = self.transform(images)
            # images = torch.unsqueeze(images, dim=0)
            sample = {"images": images, "targets": {}}

        if self.root_dir is not None:
            print("Feature Not Implemented.")
            exit(1)
            if torch.is_tensor(key):
                key = int(key[0])
            
            # load image
            img_fname = f"input_{key}.jpg"
            img_pth = os.path.join(self.root_dir, img_fname)
            image = io.imread(img_name)
            # load bb_gts

            # image_name = f"input_{key}.jpg"
            # image_name = f'bb_{idx}.jpg'
            # img_name = os.path.join(self.root_dir, image_name)
            # image = io.imread(img_name)
            # # Label order: Calorie, Protein, Fat, Sugar, Carb
            # label = torch.zeros(5)
            # nutrition = self.nutritions[image_name]
            # label[0] = nutrition["calories"]
            # label[1] = nutrition["protein"]
            # label[2] = nutrition["fat"]
            # label[3] = nutrition["sugars"]
            # label[4] = nutrition["carbohydrates"]

            # if self.transform:
            # 	image = self.transform(image)
            # sample = {'image': image, 'nutrition': label}

        return sample