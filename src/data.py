"""Datasets and dataloader factories for State Farm."""
from pathlib import Path
from typing import List, Optional, Tuple, Callable

import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold

from .config import SEED, N_FOLDS, PRETRAIN_IMG_SIZE, get_data_root
from .seed_utils import make_generator, worker_init_fn


def _fast_jpeg_open(path: str, draft_size: int) -> Image.Image:
    """JPEG decode at reduced DCT scale via libjpeg draft mode. 2-4x faster
    than full decode when target res << source res. Must be called before
    any pixel access (load/convert)."""
    img = Image.open(path)
    img.draft("RGB", (draft_size, draft_size))
    return img.convert("RGB")


def load_driver_table(data_root: Optional[Path] = None) -> pd.DataFrame:
    """Load driver_imgs_list.csv with columns: subject, classname, img."""
    data_root = data_root or get_data_root()
    csv_path = data_root / "driver_imgs_list.csv"
    df = pd.read_csv(csv_path)
    df["classname"] = df["classname"].astype(str)
    df["label"] = df["classname"].str[1].astype(int)  # 'c3' -> 3
    df["img_path"] = df.apply(
        lambda r: str(data_root / "imgs" / "train" / r["classname"] / r["img"]),
        axis=1,
    )
    return df


def list_test_images(data_root: Optional[Path] = None) -> List[str]:
    data_root = data_root or get_data_root()
    test_dir = data_root / "imgs" / "test"
    return sorted(str(p) for p in test_dir.glob("*.jpg"))


def list_train_images(data_root: Optional[Path] = None) -> List[str]:
    data_root = data_root or get_data_root()
    train_dir = data_root / "imgs" / "train"
    return sorted(str(p) for p in train_dir.rglob("*.jpg"))


def build_group_kfold(
    df: pd.DataFrame, n_splits: int = N_FOLDS
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """GroupKFold by subject. Deterministic — GroupKFold doesn't take random_state,
    but order is determined by group ordering, so seed-set numpy beforehand."""
    gkf = GroupKFold(n_splits=n_splits)
    indices = np.arange(len(df))
    return list(gkf.split(indices, df["label"].values, groups=df["subject"].values))


class UnlabeledImageDataset(Dataset):
    """For SimCLR pretrain: returns (view1, view2) per image."""

    def __init__(self, image_paths: List[str], view_generator: Callable):
        self.paths = image_paths
        self.view_generator = view_generator

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        # 2x draft target: RandomResizedCrop scale_min=0.5 -> need 2x crop size
        img = _fast_jpeg_open(self.paths[idx], PRETRAIN_IMG_SIZE * 2)
        return self.view_generator(img)


class LabeledImageDataset(Dataset):
    """For fine-tune: returns (image, label)."""

    def __init__(self, image_paths: List[str], labels: List[int], transform: Callable):
        self.paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), int(self.labels[idx])


class TestImageDataset(Dataset):
    """For test submission: returns (image_tensor, filename)."""

    def __init__(self, image_paths: List[str], transform: Callable):
        self.paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), Path(path).name


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 4,
    drop_last: bool = False,
    sampler=None,
    prefetch_factor: int = 4,
) -> DataLoader:
    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        worker_init_fn=worker_init_fn,
        generator=make_generator(SEED),
        sampler=sampler,
        persistent_workers=num_workers > 0,
    )
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**kwargs)
