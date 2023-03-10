import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import random
from torch.utils.data.dataloader import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


class SewageDataset(Dataset):
    def __init__(self, image_dir, radio=None, train=True, test=False, transform=None, seed=10086):
        if radio is None:
            radio = [0.7, 0.2, 0.1]
        self.image_dir = image_dir
        self.transform = transform
        self.images = []
        classes = os.listdir(self.image_dir)
        for ficlass in classes:
            class_path = os.path.join(self.image_dir, ficlass)
            for image in [x for x in os.listdir(class_path) if x.endswith(".jpg") or x.endswith(".jpeg")]:
                self.images.append(os.path.join(class_path, image))
        random.seed(seed)
        random.shuffle(self.images)
        self.len = len(self.images)
        if train:
            self.images = self.images[:int(radio[0] * self.len)]
        elif test:
            self.images = self.images[int(radio[2] * self.len):]
        else:
            self.images = self.images[int(radio[0] * self.len):int(radio[0] * self.len)+int(radio[1] * self.len)]
        self.len = len(self.images)

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        img_path = self.images[item]
        image = Image.open(img_path)
        image = np.array(image)

        if self.transform is not None:
            augmentations = self.transform(image=image)
            image = augmentations["image"]
        else:
            raise ValueError("Transformer is None.")

        mask = float(img_path.split("\\")[-2])

        return image, mask


def get_loaders(image_dir,
                batch_size,
                img_shape,
                num_workers=0,
                pin_memory=True,
                radio=None,
                train_transform=None,
                val_transform=None,
                ):
    if radio is None:
        radio = [0.7, 0.2, 0.1]
    if train_transform is None:
        train_transform = A.Compose(
            [
                A.Resize(height=img_shape[0], width=img_shape[1]),
                A.Rotate(limit=35, p=1.0),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.1),
                A.Normalize(
                    mean=[0.5330, 0.5463, 0.5493],
                    std=[0.1143, 0.1125, 0.1007],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ],
        )
    if val_transform is None:
        val_transform = A.Compose(
            [
                A.Resize(height=img_shape[0], width=img_shape[1]),
                A.Normalize(
                    mean=[0.5330, 0.5463, 0.5493],
                    std=[0.1143, 0.1125, 0.1007],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ],
        )
    train_dataset = SewageDataset(image_dir, radio=radio, train=True, test=False, transform=train_transform)
    val_dataset = SewageDataset(image_dir, radio=radio, train=False, test=False, transform=val_transform)
    test_dataset = SewageDataset(image_dir, radio=radio, train=False, test=True, transform=val_transform)

    train_loader = DataLoader(train_dataset,
                              batch_size,
                              shuffle=True,
                              pin_memory=pin_memory,
                              num_workers=num_workers)

    val_loader = DataLoader(val_dataset,
                            batch_size,
                            shuffle=True,
                            pin_memory=pin_memory,
                            num_workers=num_workers)

    test_loader = DataLoader(test_dataset,
                             batch_size,
                             shuffle=True,
                             pin_memory=pin_memory,
                             num_workers=num_workers)

    return train_loader, val_loader, test_loader
