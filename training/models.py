# Copyright (c) 2020 UATC LLC
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import torch
import torchvision
from torchvision.models.resnet import model_urls
from models import resnet18, resnet34, vgg, googlenet


def get_resnet18(pretrained: bool = False, **kwargs) -> torch.nn.Module:
    """Get PyTorch's default ResNet-18 model"""
    # Hack to fix SSL error while loading pretrained model -- see https://github.com/pytorch/pytorch/issues/2271
    model_urls["resnet18"] = model_urls["resnet18"].replace("https://", "http://")
    # model = torchvision.models.resnet18(pretrained=pretrained, **kwargs)
    model = resnet18(pretrained=pretrained, **kwargs)
    model._arch = "resnet18"
    return model


def get_resnet34(pretrained: bool = False, **kwargs) -> torch.nn.Module:
    model_urls["resnet34"] = model_urls["resnet34"].replace("https://", "http://")
    # model = torchvision.models.resnet34(pretrained=pretrained, **kwargs)
    model = resnet34(pretrained=pretrained, **kwargs)
    model._arch = "resnet34"
    return model


def get_resnet50(pretrained: bool = False, **kwargs) -> torch.nn.Module:
    """Get PyTorch's default ResNet-50 model"""
    model_urls["resnet50"] = model_urls["resnet50"].replace("https://", "http://")
    model = torchvision.models.resnet50(pretrained=pretrained, **kwargs)
    model._arch = "resnet50"
    return model


def get_resnet50ssl(**kwargs) -> torch.nn.Module:
    """Get a ResNet-50 pre-trained on YFC100M"""
    # Avoid SSL error due to missing python certificates -- see https://stackoverflow.com/a/60671292/1884420
    import ssl

    previous_ssl_context = ssl._create_default_https_context
    ssl._create_default_https_context = ssl._create_unverified_context

    model = torch.hub.load("facebookresearch/semi-supervised-ImageNet1K-models", "resnet50_ssl", **kwargs)
    model._arch = "resnet50"

    ssl._create_default_https_context = previous_ssl_context
    return model


def get_mask_rcnn(pretrained: bool = False, **kwargs) -> torch.nn.Module:
    """Get a mask-RCNN pre-trained on ImageNet and CoCo"""
    torchvision.models.detection.mask_rcnn.model_urls[
        "maskrcnn_resnet50_fpn_coco"
    ] = "http://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth"

    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=pretrained, **kwargs)
    return model


def get_custom_model(path: str):
    model = torch.load(path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return model


def get_uncompressed_model(
        arch: str,
        pretrained: Optional[bool] = True,
        path: Optional[str] = None,
        **kwargs) -> torch.nn.Module:
    """Gets an uncompressed network

    Parameters:
        arch: Model architecture name
        pretrained: Whether to get the pretrained model
        path: the path of custom model
    Returns:
        model: Uncompressed network
    """
    if arch == "resnet18":
        model = get_resnet18(pretrained, **kwargs)
    elif arch == "resnet34":
        model = get_resnet34(pretrained, **kwargs)
    elif arch == "resnet50":
        model = get_resnet50(pretrained, **kwargs)
    elif arch == "resnet50ssl":
        model = get_resnet50ssl(**kwargs)
    elif arch == "maskrcnn":
        model = get_mask_rcnn(pretrained, **kwargs)
    elif arch == "custom" and (path is not None):
        model = get_custom_model(path)
    elif arch == "vgg":
        assert "dataset" in kwargs.keys()
        model = vgg(**kwargs)
    elif arch == "googlenet":
        assert "num_classes" in kwargs.keys() and "aux_logits" in kwargs.keys()
        model = googlenet(pretrained, **kwargs)
    else:
        raise ValueError(f"Unknown model arch: {arch}")

    return model
