import os
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
import random


class Chinene_Medicine(Dataset):
    def __init__(self, image_dir, transform=None, seed=10086):
        self.image_dir = image_dir
        self.transform = transform
        self.images = []
        self.name_dict = {"baihe": 0, "dangshen": 1, "gouqi": 2, "huaihua": 3, "jinyinhua": 4}
        classes = os.listdir(self.image_dir)
        for ficlass in classes:
            class_path = os.path.join(self.image_dir, ficlass)
            for image in [x for x in os.listdir(class_path) if x.endswith(".jpg") or x.endswith(".jpeg")]:
                self.images.append(os.path.join(class_path, image))
        random.seed(seed)
        random.shuffle(self.images)
        self.len = len(self.images)

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        img_path = self.images[item]
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

        mask = self.name_dict[img_path.split("\\")[-2]]

        return image, mask