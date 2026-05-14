"""Day 2-4: SimCLR pretraining on Kaggle T4 x2.

Paste this into a Kaggle notebook with Accelerator = GPU T4 x2.

Strategy:
- Use DDP via torch.multiprocessing.spawn (notebook-friendly alternative to torchrun).
- 200 epochs total. Save every 10. Diagnostic every 10 (linear probe + align/uniform).
- Each session: ~5h, runs ~80 epochs at 224x224, batch 256.
- Resume from previous checkpoint by setting RESUME below.

Day 2: session 1, epochs 0-80
Day 3: resume from ep080 ckpt, run to ep160
Day 4 AM: resume from ep160 ckpt, run to ep200 (~3h), then fine-tune

After session ends, download `simclr_resnet18_latest.pth` and upload it as a
Kaggle Dataset so next session can attach it as input.

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


# Set RESUME to checkpoint path from previous session (or None for fresh start)
RESUME = None  # e.g. "/kaggle/input/simclr-ckpt-day2/simclr_resnet18_latest.pth"


def make_args(resume=None):
    return argparse.Namespace(
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


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"Detected {world_size} GPU(s)")
    args = make_args(resume=RESUME)

    if world_size > 1:
        mp.spawn(ddp_worker, args=(world_size, args), nprocs=world_size, join=True)
    else:
        run_pretrain(args)
