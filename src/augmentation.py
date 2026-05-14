"""SimCLR pretrain + supervised fine-tune augmentation pipelines.

NEVER add RandomHorizontalFlip — left/right hand is class-defining for State Farm.

Pretrain pipeline runs on uint8 CHW tensors via `torchvision.transforms.v2`,
fed from the pre-decoded memmap cache (no PIL in the hot loop).
"""
from typing import Tuple
import torch
import torchvision.transforms as T
from torchvision.transforms import v2
from PIL import Image

from .config import (
    IMG_SIZE, PRETRAIN_IMG_H, PRETRAIN_IMG_W, PRETRAIN_BLUR_KERNEL,
    IMAGENET_MEAN, IMAGENET_STD,
)


def build_pretrain_transform(size=None, blur_kernel: int = None) -> v2.Compose:
    """SimCLR augmentation in tensor space (v2). Accepts uint8 CHW tensor OR PIL.

    Defaults to T4 rectangular 120x160 + kernel=15. TPU path calls with
    size=224 + blur_kernel=23.

    Output: float32 CHW tensor, normalized. `v2.ToImage()` upfront makes the
    pipeline work for both the TPU memmap path (already CHW uint8) and the
    CUDA path (PIL via `_fast_jpeg_open`).
    """
    if size is None:
        size = (PRETRAIN_IMG_H, PRETRAIN_IMG_W)
    if blur_kernel is None:
        blur_kernel = PRETRAIN_BLUR_KERNEL
    return v2.Compose([
        v2.ToImage(),  # no-op for tensors; PIL -> uint8 CHW tensor
        v2.RandomResizedCrop(size, scale=(0.5, 1.0), antialias=True),
        v2.RandomApply([v2.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        v2.RandomGrayscale(p=0.2),
        v2.RandomApply([v2.GaussianBlur(kernel_size=blur_kernel, sigma=(0.1, 2.0))], p=0.5),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_finetune_train_transform() -> T.Compose:
    """Lighter aug for supervised fine-tune. NO horizontal flip."""
    return T.Compose([
        T.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
        T.ColorJitter(0.2, 0.2, 0.2),
        T.RandomApply([T.GaussianBlur(kernel_size=5)], p=0.3),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_eval_transform() -> T.Compose:
    """Deterministic eval transform (val + test) at IMG_SIZE (224)."""
    return T.Compose([
        T.Resize(int(IMG_SIZE * 256 / 224)),
        T.CenterCrop(IMG_SIZE),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_pretrain_eval_transform(size=None) -> T.Compose:
    """Eval transform at pretrain res. Used by linear probe.

    Defaults to T4 rectangular (H, W). Pass `size=224` for TPU path.
    """
    if size is None:
        size = (PRETRAIN_IMG_H, PRETRAIN_IMG_W)
    short = size if isinstance(size, int) else min(size)
    resize_short = int(short * 256 / 224)
    return T.Compose([
        T.Resize(resize_short),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_tta_transforms() -> list:
    """5-crop TTA. NO horizontal flip."""
    resize = int(IMG_SIZE * 256 / 224)
    base_resize = T.Resize(resize)
    normalize = T.Compose([T.ToTensor(), T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])

    def make(crop):
        return T.Compose([base_resize, crop, normalize])

    return [
        make(T.CenterCrop(IMG_SIZE)),
        make(T.Lambda(lambda img: img.crop((0, 0, IMG_SIZE, IMG_SIZE)))),
        make(T.Lambda(lambda img: img.crop((img.width - IMG_SIZE, 0, img.width, IMG_SIZE)))),
        make(T.Lambda(lambda img: img.crop((0, img.height - IMG_SIZE, IMG_SIZE, img.height)))),
        make(T.Lambda(lambda img: img.crop(
            (img.width - IMG_SIZE, img.height - IMG_SIZE, img.width, img.height)
        ))),
    ]


class ContrastiveViewGenerator:
    """Apply transform twice to produce a positive pair for SimCLR.

    Works for both PIL inputs (legacy) and uint8 CHW tensors (v2 pipeline).
    """

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img) -> Tuple:
        return self.transform(img), self.transform(img)
