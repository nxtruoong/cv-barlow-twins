"""SimCLR pretraining on Kaggle TPU v5e-8 (PyTorch/XLA).

Entry:
  python -m src.pretrain_xla [--epochs ...] [...]

Launches `xmp.spawn` with one process per TPU chip (8 on v5e-8). Each rank
reads from the shared pre-decoded memmap cache (build it first via
`python -m src.cache_decoder`).

Differences from `src/pretrain.py` (CUDA):
- No GradScaler, no cudnn flags, no channels_last, no torch.compile.
- bf16 autocast on forward + loss; weights/optimizer stay fp32.
- Distributed = xmp.spawn + xm.optimizer_step (gradient all-reduce).
- NT-Xent gather goes through the XLA differentiable path in `loss.py`.
- DataLoader is wrapped in `MpDeviceLoader` for host->device overlap.
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from timm.scheduler import CosineLRScheduler

import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .augmentation import build_pretrain_transform, ContrastiveViewGenerator
from .config import (
    PRETRAIN_PER_CORE_BATCH, PRETRAIN_EPOCHS, PRETRAIN_WARMUP_EPOCHS,
    PRETRAIN_LR, PRETRAIN_WEIGHT_DECAY, get_working_dir,
)
from .data import MemmapUnlabeledDataset
from .loss import NTXentLoss
from .model import SimCLRModel
from .seed_utils import set_seed


def _save_checkpoint(model: nn.Module, optimizer, scheduler, epoch: int,
                     history: list, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # xm.save handles host transfer + atomic write; only rank 0 actually writes.
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "history": history,
    }
    xm.save(state, str(out_dir / f"simclr_resnet18_ep{epoch:03d}.pth"))
    xm.save(state, str(out_dir / "simclr_resnet18_latest.pth"))


def _train_one_epoch(model, loader, optimizer, loss_fn, device, epoch: int,
                     rank: int) -> dict:
    model.train()
    loss_accum = torch.zeros((), device=device)
    n_batches = 0
    epoch_t0 = time.time()

    for v1, v2 in loader:
        with torch.autocast("xla", dtype=torch.bfloat16):
            v = torch.cat([v1, v2], dim=0)
            z = model(v)
            z1, z2 = z.chunk(2, dim=0)
            loss = loss_fn(z1, z2)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        xm.optimizer_step(optimizer)  # all-reduce grads, then optimizer.step()

        loss_accum = loss_accum + loss.detach()
        n_batches += 1

    # Single host sync at epoch end.
    avg_loss = (loss_accum / max(n_batches, 1)).item()
    epoch_sec = time.time() - epoch_t0
    return {"loss": avg_loss, "epoch_sec": epoch_sec, "n_batches": n_batches}


def _mp_main(rank: int, args: argparse.Namespace) -> None:
    set_seed()
    device = xm.xla_device()
    world_size = xm.xrt_world_size()

    if xm.is_master_ordinal():
        print(f"World size: {world_size}, per-core batch: {args.per_core_batch}, "
              f"global batch: {world_size * args.per_core_batch}")

    view_gen = ContrastiveViewGenerator(build_pretrain_transform())
    dataset = MemmapUnlabeledDataset(view_generator=view_gen)

    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=xm.get_ordinal(),
        shuffle=True, seed=24521897, drop_last=True,
    )
    raw_loader = DataLoader(
        dataset, batch_size=args.per_core_batch, sampler=sampler,
        num_workers=args.num_workers, drop_last=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )

    model = SimCLRModel(pretrained_backbone=False).to(device)
    loss_fn = NTXentLoss(gather_distributed=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr * (world_size * args.per_core_batch / 256) ** 0.5
            if args.scale_lr else args.lr,
        weight_decay=PRETRAIN_WEIGHT_DECAY,
    )
    scheduler = CosineLRScheduler(
        optimizer, t_initial=args.epochs, warmup_t=PRETRAIN_WARMUP_EPOCHS,
        warmup_lr_init=1e-6, lr_min=0.0,
    )

    start_epoch = 0
    history: list = []
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if ckpt.get("scheduler_state_dict"):
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        history = ckpt.get("history", [])
        if xm.is_master_ordinal():
            print(f"Resumed from {args.resume} at epoch {start_epoch}")

    out_dir = Path(args.output_dir) if args.output_dir else get_working_dir() / "simclr"

    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        scheduler.step(epoch)
        loader = pl.MpDeviceLoader(raw_loader, device)

        stats = _train_one_epoch(
            model, loader, optimizer, loss_fn, device, epoch, xm.get_ordinal(),
        )

        lr = optimizer.param_groups[0]["lr"]
        log_entry = {"epoch": epoch, "lr": lr, **stats}
        history.append(log_entry)

        if xm.is_master_ordinal():
            print(f"[ep {epoch}] loss={stats['loss']:.4f} "
                  f"sec={stats['epoch_sec']:.1f} lr={lr:.2e}")

        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            _save_checkpoint(model, optimizer, scheduler, epoch, history, out_dir)
            if xm.is_master_ordinal():
                with open(out_dir / "history.json", "w") as f:
                    json.dump(history, f, indent=2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=PRETRAIN_EPOCHS)
    p.add_argument("--per-core-batch", type=int, default=PRETRAIN_PER_CORE_BATCH)
    p.add_argument("--lr", type=float, default=PRETRAIN_LR)
    p.add_argument("--scale-lr", action="store_true", default=False,
                   help="re-scale LR by sqrt(global_batch/256) at startup")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--nprocs", type=int, default=8, help="TPU chips (8 on v5e-8)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    xmp.spawn(_mp_main, args=(args,), nprocs=args.nprocs, start_method="fork")


if __name__ == "__main__":
    main()
