"""SimCLR pretraining loop. Single-GPU or DDP (launched via torchrun)."""
import argparse
import json
import os
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
    build_eval_transform
from .config import (
    PRETRAIN_BATCH_SIZE, PRETRAIN_EPOCHS, PRETRAIN_WARMUP_EPOCHS,
    PRETRAIN_LR, PRETRAIN_WEIGHT_DECAY, get_working_dir,
)
from .data import (
    UnlabeledImageDataset, LabeledImageDataset, list_train_images,
    list_test_images, load_driver_table, build_group_kfold, make_loader,
)
from .diagnostics import linear_probe, sample_alignment_uniformity
from .loss import NTXentLoss
from .model import SimCLRModel
from .seed_utils import set_seed


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
        "model_state_dict": (model.module if hasattr(model, "module") else model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "history": history,
    }
    path = out_dir / f"simclr_resnet18_ep{epoch:03d}.pth"
    torch.save(state, path)
    latest = out_dir / "simclr_resnet18_latest.pth"
    torch.save(state, latest)


def build_probe_loaders(batch_size: int, num_workers: int) -> tuple:
    df = load_driver_table()
    folds = build_group_kfold(df)
    train_idx, val_idx = folds[0]
    eval_tf = build_eval_transform()

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
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    pretrain_paths = list_train_images()
    test_paths = list_test_images()
    all_paths = pretrain_paths + test_paths
    if is_main(rank):
        print(f"Pretrain dataset: {len(all_paths)} images "
              f"({len(pretrain_paths)} train + {len(test_paths)} test)")

    view_gen = ContrastiveViewGenerator(build_pretrain_transform())
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

    model = SimCLRModel(pretrained_backbone=False).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False)

    loss_fn = NTXentLoss(gather_distributed=(world_size > 1))

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=PRETRAIN_WEIGHT_DECAY,
    )
    scheduler = CosineLRScheduler(
        optimizer, t_initial=args.epochs, warmup_t=PRETRAIN_WARMUP_EPOCHS,
        warmup_lr_init=1e-6, lr_min=0.0,
    )
    scaler = GradScaler("cuda", enabled=args.amp)

    start_epoch = 0
    history = []
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location="cpu")
        target = model.module if hasattr(model, "module") else model
        target.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if ckpt.get("scheduler_state_dict"):
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        history = ckpt.get("history", [])
        if is_main(rank):
            print(f"Resumed from {args.resume} at epoch {start_epoch}")

    out_dir = Path(args.output_dir) if args.output_dir else get_working_dir() / "simclr"

    for epoch in range(start_epoch, args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        scheduler.step(epoch)

        running = 0.0
        n_batches = 0
        pbar = tqdm(loader, disable=not is_main(rank), desc=f"epoch {epoch}")
        for v1, v2 in pbar:
            v1 = v1.to(device, non_blocking=True)
            v2 = v2.to(device, non_blocking=True)

            # Concat views into single forward pass. Two separate forwards
            # through a DDP-wrapped model corrupts autograd version counters
            # ("variable modified by inplace op" error). Also faster.
            with autocast("cuda", enabled=args.amp):
                v = torch.cat([v1, v2], dim=0)
                z = model(v)
                z1, z2 = z.chunk(2, dim=0)
                loss = loss_fn(z1, z2)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running += loss.item()
            n_batches += 1
            if is_main(rank):
                pbar.set_postfix(loss=running / n_batches)

        epoch_loss = running / max(n_batches, 1)
        log_entry = {"epoch": epoch, "loss": epoch_loss}

        if is_main(rank) and (epoch + 1) % args.diagnostic_every == 0:
            base = model.module if hasattr(model, "module") else model
            base.eval()
            au = sample_alignment_uniformity(base, loader, device, max_batches=3)
            probe_train, probe_val = build_probe_loaders(
                batch_size=256, num_workers=args.num_workers,
            )
            probe = linear_probe(base.backbone, probe_train, probe_val, device)
            log_entry.update(au)
            log_entry.update(probe)
            print(f"[ep {epoch}] loss={epoch_loss:.4f} "
                  f"align={au['alignment']:.4f} uniform={au['uniformity']:.4f} "
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
    """DDP entry for torch.multiprocessing.spawn (notebook-friendly).

    Must live at module level so spawn child processes can import it by name.
    """
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    run_pretrain(args)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=PRETRAIN_EPOCHS)
    p.add_argument("--batch-size", type=int, default=PRETRAIN_BATCH_SIZE)
    p.add_argument("--lr", type=float, default=PRETRAIN_LR)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--diagnostic-every", type=int, default=10)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    run_pretrain(parse_args())
