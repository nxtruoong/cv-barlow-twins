"""Day 2-4: SimCLR pretraining on Kaggle T4 x2.

Paste this into a Kaggle notebook with Accelerator = GPU T4 x2.

Strategy:
- Use DDP via torch.multiprocessing.spawn (notebook-friendly alternative to torchrun).
- 200 epochs total. Save every 10. Diagnostic every 10 (linear probe + align/uniform).
- Each session: ~5h, runs ~80 epochs at 224x224, batch 256.
- Resume from previous checkpoint by passing --resume.

Day 2: session 1, epochs 0-80
Day 3: resume from ep080 ckpt, run to ep160
Day 4 AM: resume from ep160 ckpt, run to ep200 (~3h), then fine-tune

After session ends, download `simclr_resnet18_latest.pth` and upload it as a
Kaggle Dataset so next session can attach it as input.
"""
import os
import sys
import torch
import torch.multiprocessing as mp

sys.path.insert(0, "/kaggle/working/CV")

from src.pretrain import run_pretrain
import argparse


def make_args(resume=None):
    args = argparse.Namespace(
        epochs=200,
        batch_size=256,
        lr=1e-3,
        num_workers=4,
        amp=True,
        save_every=10,
        diagnostic_every=10,
        resume=resume,
        output_dir="/kaggle/working/simclr",
    )
    return args


def ddp_worker(rank: int, world_size: int, resume: str):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    run_pretrain(make_args(resume=resume))


if __name__ == "__main__":
    # Set RESUME to checkpoint path from previous session (or None for fresh start)
    RESUME = None  # e.g. "/kaggle/input/simclr-ckpt-day2/simclr_resnet18_latest.pth"

    world_size = torch.cuda.device_count()
    print(f"Detected {world_size} GPU(s)")

    if world_size > 1:
        mp.spawn(ddp_worker, args=(world_size, RESUME), nprocs=world_size, join=True)
    else:
        run_pretrain(make_args(resume=RESUME))
