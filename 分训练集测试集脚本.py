import os
import random
import shutil
from typing import List


def main(path: str, seed: int = 10086, radio: List = None):
    if radio is None:
        radio = [0.7, 0.3]
    assert len(radio) == 2

    parent_path = os.path.dirname(path)
    parent_path = os.path.abspath(parent_path)
    train_path = os.path.join(parent_path, "train")
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    val_path = os.path.join(parent_path, "val")
    if not os.path.exists(val_path):
        os.makedirs(val_path)

    for classes in os.listdir(path):
        print(classes)
        classes_path = os.path.join(path, classes)
        train_class_path = os.path.join(train_path, classes)
        if not os.path.exists(train_class_path):
            os.makedirs(train_class_path)
        val_class_path = os.path.join(val_path, classes)
        if not os.path.exists(val_class_path):
            os.makedirs(val_class_path)
        img_names = os.listdir(classes_path)
        random.seed(seed)
        random.shuffle(img_names)
        radio_i = radio[0]/sum(radio)
        index = int(len(img_names) * radio_i)
        train_imgs = img_names[:index]
        val_imgs = img_names[index:]
        for img in train_imgs:
            img_path = os.path.join(classes_path, img)
            img_copy_path = os.path.join(train_path, classes, img)
            shutil.copy(img_path, img_copy_path)
        for img in val_imgs:
            img_path = os.path.join(classes_path, img)
            img_copy_path = os.path.join(val_path, classes, img)
            shutil.copy(img_path, img_copy_path)


if __name__ == "__main__":
    main(r"E:\LY\data\ChineseMedicine\Chinese Medicine")
