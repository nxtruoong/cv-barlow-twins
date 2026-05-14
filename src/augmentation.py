"""SimCLR pretrain + supervised fine-tune augmentation pipelines.

NEVER add RandomHorizontalFlip — left/right hand is class-defining for State Farm.
"""
from typing import Tuple
import torchvision.transforms as T
from PIL import Image

from .config import IMG_SIZE, PRETRAIN_IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD


def build_pretrain_transform() -> T.Compose:
    """SimCLR augmentation. Tuned for State Farm cabin scenes (not ImageNet).

    Pretrain at PRETRAIN_IMG_SIZE (160) for ~2x FLOP cut vs 224. Phase 2
    fine-tune upscales back to IMG_SIZE (224) — FixRes trick.
    Blur kernel scaled to ~10% of pretrain resolution (15 px for 160).
    """
    return T.Compose([
        T.RandomResizedCrop(PRETRAIN_IMG_SIZE, scale=(0.5, 1.0)),
        T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.RandomApply([T.GaussianBlur(kernel_size=15, sigma=(0.1, 2.0))], p=0.5),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_finetune_train_transform() -> T.Compose:
    """Lighter aug for supervised fine-tune."""
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


def build_pretrain_eval_transform() -> T.Compose:
    """Eval transform at PRETRAIN_IMG_SIZE (160). Use for linear probe during
    SimCLR pretraining so probe res matches pretrain res."""
    resize = int(PRETRAIN_IMG_SIZE * 256 / 224)
    return T.Compose([
        T.Resize(resize),
        T.CenterCrop(PRETRAIN_IMG_SIZE),
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
    """Apply transform twice to produce a positive pair for SimCLR."""

    def __init__(self, transform: T.Compose):
        self.transform = transform

    def __call__(self, img: Image.Image) -> Tuple:
        return self.transform(img), self.transform(img)
