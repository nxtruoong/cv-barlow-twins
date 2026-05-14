"""One-shot JPEG -> uint8 memmap pre-decoder for SimCLR pretrain on TPU.

Output:
  PRETRAIN_CACHE_PATH  = (N, R, R, 3) uint8 raw bytes (np.memmap)
  PRETRAIN_CACHE_INDEX = {"n": int, "res": int, "paths": [str, ...]}

Run once per Kaggle session:
  python -m src.cache_decoder

Re-runs idempotent: skips if output exists and N matches input listing.
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
from tqdm import tqdm

from .config import (
    PRETRAIN_CACHE_RES, PRETRAIN_CACHE_PATH, PRETRAIN_CACHE_INDEX,
)
from .data import list_train_images, list_test_images


def _decode_one(args) -> int:
    idx, path, res, mm_path, n = args
    img = Image.open(path)
    img.draft("RGB", (res * 2, res * 2))  # libjpeg fast path
    img = img.convert("RGB").resize((res, res), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.uint8)  # (R, R, 3)
    # Each worker reopens memmap to keep file handles per-thread safe.
    mm = np.memmap(mm_path, dtype=np.uint8, mode="r+",
                   shape=(n, res, res, 3))
    mm[idx] = arr
    del mm
    return idx


def build_cache(
    paths: List[str],
    res: int = PRETRAIN_CACHE_RES,
    mm_path: str = PRETRAIN_CACHE_PATH,
    index_path: str = PRETRAIN_CACHE_INDEX,
    workers: int = 16,
) -> None:
    n = len(paths)
    expected_bytes = n * res * res * 3

    if (Path(mm_path).exists() and Path(index_path).exists()
            and Path(mm_path).stat().st_size == expected_bytes):
        with open(index_path) as f:
            meta = json.load(f)
        if meta.get("n") == n and meta.get("res") == res:
            print(f"Cache exists with N={n}, res={res} — skipping rebuild.")
            return

    Path(mm_path).parent.mkdir(parents=True, exist_ok=True)
    # Allocate sparse file of correct size.
    with open(mm_path, "wb") as f:
        f.truncate(expected_bytes)

    args_iter = [(i, p, res, mm_path, n) for i, p in enumerate(paths)]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for _ in tqdm(ex.map(_decode_one, args_iter), total=n, desc="decode"):
            pass

    with open(index_path, "w") as f:
        json.dump({"n": n, "res": res, "paths": paths}, f)
    print(f"Cache built: {mm_path} ({expected_bytes / 1e9:.1f} GB)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--res", type=int, default=PRETRAIN_CACHE_RES)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--out", type=str, default=PRETRAIN_CACHE_PATH)
    p.add_argument("--index", type=str, default=PRETRAIN_CACHE_INDEX)
    args = p.parse_args()

    paths = list_train_images() + list_test_images()
    build_cache(paths, args.res, args.out, args.index, args.workers)


if __name__ == "__main__":
    main()
