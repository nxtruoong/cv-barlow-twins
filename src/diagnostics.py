"""SSL diagnostics: linear probe + alignment/uniformity tracking."""
from typing import Dict, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import NUM_CLASSES


@torch.no_grad()
def extract_features(
    backbone: nn.Module, loader: DataLoader, device: torch.device
) -> tuple:
    backbone.eval()
    feats, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        h = backbone(x)
        feats.append(h.cpu())
        labels.append(y)
    return torch.cat(feats, dim=0), torch.cat(labels, dim=0)


def linear_probe(
    backbone: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> Dict[str, float]:
    """Freeze backbone, train Linear(feat_dim -> 10), report val accuracy + log loss."""
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    train_feats, train_labels = extract_features(backbone, train_loader, device)
    val_feats, val_labels = extract_features(backbone, val_loader, device)

    feat_dim = train_feats.size(1)
    head = nn.Linear(feat_dim, NUM_CLASSES).to(device)
    optim = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    train_feats = train_feats.to(device)
    train_labels = train_labels.to(device)

    batch_size = 512
    n = train_feats.size(0)

    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            logits = head(train_feats[idx])
            loss = loss_fn(logits, train_labels[idx])
            optim.zero_grad()
            loss.backward()
            optim.step()

    head.eval()
    val_feats = val_feats.to(device)
    val_labels = val_labels.to(device)
    with torch.no_grad():
        logits = head(val_feats)
        probs = F.softmax(logits, dim=1)
        pred = logits.argmax(dim=1)
        acc = (pred == val_labels).float().mean().item()
        ll = F.nll_loss(torch.log(probs.clamp_min(1e-15)), val_labels).item()

    return {"linear_probe_acc": acc, "linear_probe_log_loss": ll}


@torch.no_grad()
def sample_alignment_uniformity(
    simclr_model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 5,
) -> Dict[str, float]:
    """Estimate align/uniform from a few pretrain batches."""
    from .loss import alignment_uniformity

    simclr_model.eval()
    aligns, uniforms = [], []
    for i, (v1, v2) in enumerate(loader):
        if i >= max_batches:
            break
        v1 = v1.to(device, non_blocking=True)
        v2 = v2.to(device, non_blocking=True)
        z1 = simclr_model(v1)
        z2 = simclr_model(v2)
        stats = alignment_uniformity(z1, z2)
        aligns.append(stats["alignment"])
        uniforms.append(stats["uniformity"])

    return {
        "alignment": float(np.mean(aligns)),
        "uniformity": float(np.mean(uniforms)),
    }
