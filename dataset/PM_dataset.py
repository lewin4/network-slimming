import os
from abc import ABC
import pandas as pd
import random
import PIL.Image as Image

import torch

from torch.utils.data import Dataset
from torchvision import transforms
from typing import Optional, Callable


class PM_dataset(Dataset, ABC):
    def __init__(
            self,
            train_image_dir: str,
            val_image_dir: str,
            transform: Optional[Callable] = None,
            train: bool = True,
    ):
        self.transform = transform
        self.train = train
        if self.train:
            self.image_dir = train_image_dir
        else:
            self.image_dir = val_image_dir
            self.val_label = pd.read_excel(
                io=os.path.join(os.path.abspath(os.path.join(self.image_dir, os.pardir)),
                                "PM_Label_and_Fovea_Location.xlsx"),
                usecols="B:C",
            )
        self.image_list = os.listdir(self.image_dir)
        random.shuffle(self.image_list)
        self.len = len(self.image_list)

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        image_name = self.image_list[item]
        img_path = os.path.join(self.image_dir, image_name)
        image = Image.open(img_path)

        if self.transform is not None:
            image = self.transform(image)
        else:
            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                transforms.ToTensor(),
            ])
            image = transform(image)

        if self.train:
            label_name = image_name[0]
            if label_name in ["H", "N"]:
                label = torch.Tensor([0])
            elif label_name in ["P"]:
                label = torch.Tensor([1])
            else:
                raise Exception("label {} wrong!!".format(label_name))
            return image, label
        else:
            label = self.val_label[self.val_label["imgName"] == image_name]["Label"]
            label = torch.Tensor([int(label)])
            return image, label
