"""SimCLR pretraining on Kaggle T4 x2.

Paste this into a Kaggle notebook with Accelerator = GPU T4 x2.

Strategy:
- DDP via torch.multiprocessing.spawn (notebook-friendly alternative to torchrun).
- 100 epochs total, batch 768, LR sqrt-scaled, rect 160x120 (W x H).
- torch.compile(max-autotune) + FP16 AMP + channels_last. First epoch slower
  due to one-time autotune (1-3 min); steady-state ~3-4 min/epoch.
- Target: <10h wallclock single session. No multi-day resume needed.
- Save every 10. Diagnostic every 10 (linear probe + align/uniform).
- Resume from previous checkpoint by setting RESUME below (optional).

After session ends, download `simclr_resnet18_latest.pth` and upload as a
Kaggle Dataset so fine-tune notebook can attach it as input.

NOTE: ddp_worker MUST be defined in src.pretrain (importable module), not in this
notebook script. torch.multiprocessing.spawn pickles by qualified name and child
processes re-import __main__ — functions defined inside exec() in a notebook are
not picklable for spawn children.
"""
import sys
import os
import argparse

REPO_ROOT = "/kaggle/working/CV"
sys.path.insert(0, REPO_ROOT)
# Child spawn processes inherit env but NOT sys.path. Set PYTHONPATH so
# `from src.pretrain import ddp_worker` resolves inside spawned workers.
os.environ["PYTHONPATH"] = REPO_ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")

import torch
import torch.multiprocessing as mp

from src.pretrain import run_pretrain, ddp_worker
from src.config import PRETRAIN_BATCH_SIZE, PRETRAIN_EPOCHS, PRETRAIN_LR


# Set RESUME to checkpoint path from previous session (or None for fresh start)
RESUME = None  # e.g. "/kaggle/input/simclr-ckpt/simclr_resnet18_latest.pth"


def make_args(resume=None):
    return argparse.Namespace(
        epochs=PRETRAIN_EPOCHS,
        batch_size=PRETRAIN_BATCH_SIZE,
        lr=PRETRAIN_LR,
        num_workers=2,  # per-rank; DDP world=2 -> 4 total, matches Kaggle 4-CPU env
        amp=True,
        no_compile=False,
        save_every=10,
        diagnostic_every=10,
        resume=resume,
        output_dir="/kaggle/working/simclr",
    )


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"Detected {world_size} GPU(s)")
    args = make_args(resume=RESUME)

    if world_size > 1:
        mp.spawn(ddp_worker, args=(world_size, args), nprocs=world_size, join=True)
    else:
        run_pretrain(args)
