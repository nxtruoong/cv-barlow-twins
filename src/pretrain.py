"""Barlow Twins pretraining loop. Single-GPU or DDP on CUDA (T4 x2).

Phase 1 SSL method per ADR-0003 (Barlow Twins replaces SimCLR). The TPU/XLA
path used by the prior SimCLR plan lives on the ``tpu-v5e-8-snapshot``
branch and is no longer wired here.

Key requirements for correctness:

1. ``nn.SyncBatchNorm.convert_sync_batchnorm`` BEFORE wrapping in DDP, so
   the final ``BatchNorm1d(affine=False)`` in the projector reflects the
   global batch. Local-batch BN silently undertrains.
2. Single concatenated forward per step with per-view BN inside
   ``BarlowTwinsModel.forward(v1, v2)`` — avoids DDP multiple-forward
   autograd quirks observed previously with SimCLR.
3. C matrix summed across ranks via ``dist.all_reduce`` inside
   ``BarlowTwinsLoss`` so the loss reflects the global cross-correlation.
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from timm.scheduler import CosineLRScheduler
from tqdm import tqdm

from .augmentation import build_pretrain_transform, ContrastiveViewGenerator, \
    build_pretrain_eval_transform
from .config import (
    BT_BATCH_SIZE, BT_EPOCHS, BT_WARMUP_EPOCHS, BT_LR, BT_WEIGHT_DECAY,
    get_working_dir,
)
from .data import (
    UnlabeledImageDataset, LabeledImageDataset, list_train_images,
    list_test_images, load_driver_table, build_group_kfold, make_loader,
)
from .diagnostics import linear_probe
from .loss import BarlowTwinsLoss, cross_correlation_stats
from .model import BarlowTwinsModel
from .seed_utils import set_seed


# Epochs at which to run the (expensive) linear probe. Lightweight C-matrix
# stats are logged every epoch.
LINEAR_PROBE_EPOCHS = {20, 40, 60, 80}


def _unwrap(m: nn.Module) -> nn.Module:
    """Strip DDP wrapper to access underlying module."""
    return getattr(m, "module", m)


def setup_distributed() -> tuple:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def is_main(rank: int) -> bool:
    return rank == 0


def save_checkpoint(model: nn.Module, optimizer, scheduler, epoch: int,
                    history: list, out_dir: Path, rank: int) -> None:
    if not is_main(rank):
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": _unwrap(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "history": history,
    }
    path = out_dir / f"bt_resnet18_ep{epoch:03d}.pth"
    torch.save(state, path)
    latest = out_dir / "bt_resnet18_latest.pth"
    torch.save(state, latest)


def build_probe_loaders(batch_size: int, num_workers: int) -> tuple:
    df = load_driver_table()
    folds = build_group_kfold(df)
    train_idx, val_idx = folds[0]
    eval_tf = build_pretrain_eval_transform(size=224)

    train_ds = LabeledImageDataset(
        df.iloc[train_idx]["img_path"].tolist(),
        df.iloc[train_idx]["label"].tolist(),
        eval_tf,
    )
    val_ds = LabeledImageDataset(
        df.iloc[val_idx]["img_path"].tolist(),
        df.iloc[val_idx]["label"].tolist(),
        eval_tf,
    )
    train_loader = make_loader(train_ds, batch_size, shuffle=False,
                               num_workers=num_workers)
    val_loader = make_loader(val_ds, batch_size, shuffle=False,
                             num_workers=num_workers)
    return train_loader, val_loader


def run_pretrain(args) -> None:
    set_seed()
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    pretrain_paths = list_train_images()
    test_paths = list_test_images()
    all_paths = pretrain_paths + test_paths
    if is_main(rank):
        print(f"Pretrain dataset: {len(all_paths)} images "
              f"({len(pretrain_paths)} train + {len(test_paths)} test)")

    # BT runs at 224 square (paper baseline) with blur kernel 23 (~10% of width).
    # The default (120x160 rect, k=15) is a SimCLR-T4 leftover.
    view_gen = ContrastiveViewGenerator(
        build_pretrain_transform(size=224, blur_kernel=23)
    )
    dataset = UnlabeledImageDataset(all_paths, view_gen)

    per_gpu_batch = args.batch_size // max(world_size, 1)
    if world_size > 1:
        sampler = DistributedSampler(dataset, shuffle=True, seed=24521897, drop_last=True)
    else:
        sampler = None
    loader = make_loader(
        dataset, batch_size=per_gpu_batch, shuffle=(sampler is None),
        num_workers=args.num_workers, drop_last=True, sampler=sampler,
    )

    model = BarlowTwinsModel(pretrained_backbone=False).to(device)
    model = model.to(memory_format=torch.channels_last)
    if world_size > 1:
        # SyncBN BEFORE DDP wrap. Without this the final
        # BatchNorm1d(affine=False) uses per-GPU stats -> per-GPU C matrix.
        model = BarlowTwinsModel.convert_sync_bn(model)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False)

    loss_fn = BarlowTwinsLoss(lambda_off=args.lambda_off)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=BT_WEIGHT_DECAY,
    )
    scheduler = CosineLRScheduler(
        optimizer, t_initial=args.epochs, warmup_t=BT_WARMUP_EPOCHS,
        warmup_lr_init=1e-6, lr_min=0.0,
    )
    scaler = GradScaler("cuda", enabled=args.amp)

    start_epoch = 0
    history = []
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location="cpu")
        _unwrap(model).load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if ckpt.get("scheduler_state_dict"):
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        history = ckpt.get("history", [])
        if is_main(rank):
            print(f"Resumed from {args.resume} at epoch {start_epoch}")

    out_dir = Path(args.output_dir) if args.output_dir else get_working_dir() / "bt"

    for epoch in range(start_epoch, args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        scheduler.step(epoch)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        loss_accum = torch.zeros((), device=device)
        DIAG_EVERY = 20
        n_batches = 0
        diag_count = 0
        t_data_sum = 0.0
        t_compute_sum = 0.0
        diag_mean_sum = 0.0
        off_rms_sum = 0.0
        images_seen = 0
        epoch_t0 = time.time()
        pbar = tqdm(loader, disable=not is_main(rank), desc=f"epoch {epoch}")
        t_iter = time.time()
        for v1, v2 in pbar:
            t_data = time.time() - t_iter
            v1 = v1.to(device, non_blocking=True, memory_format=torch.channels_last)
            v2 = v2.to(device, non_blocking=True, memory_format=torch.channels_last)

            with autocast("cuda", enabled=args.amp):
                z1, z2 = model(v1, v2)
                loss = loss_fn(z1, z2)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_accum += loss.detach()
            n_batches += 1
            images_seen += v1.size(0) * 2
            t_compute = time.time() - t_iter - t_data
            t_data_sum += t_data
            t_compute_sum += t_compute

            if n_batches % DIAG_EVERY == 0:
                with torch.no_grad():
                    stats = cross_correlation_stats(z1, z2)
                diag_mean_sum += stats["c_diag_mean"]
                off_rms_sum += stats["c_offdiag_rms"]
                diag_count += 1
                if is_main(rank):
                    pbar.set_postfix(
                        loss=(loss_accum / n_batches).item(),
                        diag=diag_mean_sum / diag_count,
                        off=off_rms_sum / diag_count,
                    )
            t_iter = time.time()

        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_sec = time.time() - epoch_t0
        throughput = images_seen / max(epoch_sec, 1e-6)
        gpu_mem_peak_gb = (
            torch.cuda.max_memory_allocated(device) / 1e9
            if device.type == "cuda" else 0.0
        )
        current_lr = optimizer.param_groups[0]["lr"]

        if is_main(rank):
            print(f"[ep {epoch}] avg data={t_data_sum/max(n_batches,1):.3f}s "
                  f"compute={t_compute_sum/max(n_batches,1):.3f}s "
                  f"| {throughput:.0f} img/s | {epoch_sec/60:.1f} min "
                  f"| peak_mem={gpu_mem_peak_gb:.2f} GB | lr={current_lr:.2e}")

        epoch_loss = (loss_accum / max(n_batches, 1)).item()
        log_entry = {
            "epoch": epoch,
            "loss": epoch_loss,
            "c_diag_mean": diag_mean_sum / max(diag_count, 1),
            "c_offdiag_rms": off_rms_sum / max(diag_count, 1),
            "lr": current_lr,
            "throughput_img_s": throughput,
            "epoch_sec": epoch_sec,
            "gpu_mem_peak_gb": gpu_mem_peak_gb,
            "t_data_avg": t_data_sum / max(n_batches, 1),
            "t_compute_avg": t_compute_sum / max(n_batches, 1),
        }

        # Linear probe at pre-registered epochs only — too expensive to run
        # every 10. C-matrix stats above already gate against silent collapse.
        if is_main(rank) and (epoch + 1) in LINEAR_PROBE_EPOCHS:
            base = _unwrap(model)
            base.eval()
            probe_train, probe_val = build_probe_loaders(
                batch_size=256, num_workers=args.num_workers,
            )
            probe = linear_probe(base.backbone, probe_train, probe_val, device)
            log_entry.update(probe)
            print(f"[ep {epoch}] loss={epoch_loss:.4f} "
                  f"diag={log_entry['c_diag_mean']:.3f} "
                  f"off_rms={log_entry['c_offdiag_rms']:.3f} "
                  f"probe_acc={probe['linear_probe_acc']:.4f} "
                  f"probe_ll={probe['linear_probe_log_loss']:.4f}")

        history.append(log_entry)

        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            save_checkpoint(model, optimizer, scheduler, epoch, history, out_dir, rank)
            if is_main(rank):
                with open(out_dir / "history.json", "w") as f:
                    json.dump(history, f, indent=2)

    if world_size > 1:
        dist.destroy_process_group()


def ddp_worker(rank: int, world_size: int, args) -> None:
    """DDP entry for torch.multiprocessing.spawn (Kaggle notebook-friendly)."""
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    run_pretrain(args)


def parse_args() -> argparse.Namespace:
    from .config import BT_LAMBDA
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=BT_EPOCHS)
    p.add_argument("--batch-size", type=int, default=BT_BATCH_SIZE)
    p.add_argument("--lr", type=float, default=BT_LR)
    p.add_argument("--lambda-off", type=float, default=BT_LAMBDA)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    run_pretrain(parse_args())
