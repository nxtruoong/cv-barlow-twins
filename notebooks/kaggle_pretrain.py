"""Barlow Twins pretraining on Kaggle T4 x2 (Phase 1, per ADR-0003).

Paste into a Kaggle notebook with Accelerator = GPU T4 x2.

Strategy:
- DDP via torch.multiprocessing.spawn (notebook-friendly; torchrun awkward).
- 80 epochs total, batch 256 (128 per GPU), AdamW lr=1e-3, lambda=5e-3.
- SyncBatchNorm applied inside src.pretrain BEFORE DDP wrap so the final
  projector BatchNorm1d(affine=False) reflects the global batch.
- Target: ~10h wallclock. Split across 2 Kaggle sessions (5h each, 40 ep each).
- Save every 10 epochs. Linear probe at epochs 20/40/60/80 only.

Resume protocol between sessions:
1. End of session 1: download `bt_resnet18_latest.pth` from
   /kaggle/working/bt/, upload as a new Kaggle Dataset.
2. Session 2: attach that Dataset, set RESUME to its path, run.

NOTE: ddp_worker MUST live in src.pretrain (importable module). mp.spawn
pickles by qualified name; functions defined in notebook __main__ are
not picklable for spawn children.
"""
import sys
import os
import argparse

REPO_ROOT = "/kaggle/working/CV"
sys.path.insert(0, REPO_ROOT)
os.environ["PYTHONPATH"] = REPO_ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")

import torch
import torch.multiprocessing as mp

from src.pretrain import run_pretrain, ddp_worker
from src.config import BT_BATCH_SIZE, BT_EPOCHS, BT_LR, BT_LAMBDA


# Set to a checkpoint path from a previous session (or None for fresh start).
# Example: "/kaggle/input/bt-ckpt-session1/bt_resnet18_latest.pth"
RESUME = None


def make_args(resume=None):
    return argparse.Namespace(
        epochs=BT_EPOCHS,
        batch_size=BT_BATCH_SIZE,
        lr=BT_LR,
        lambda_off=BT_LAMBDA,
        num_workers=2,  # per-rank; DDP world=2 -> 4 total (Kaggle 4-CPU env)
        amp=True,
        save_every=10,
        resume=resume,
        output_dir="/kaggle/working/bt",
    )


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"Detected {world_size} GPU(s)")
    args = make_args(resume=RESUME)

    if world_size > 1:
        mp.spawn(ddp_worker, args=(world_size, args), nprocs=world_size, join=True)
    else:
        run_pretrain(args)
