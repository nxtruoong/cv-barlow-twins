"""NT-Xent loss with DDP-aware all_gather.

Critical detail: must preserve local-rank gradient. Naive `dist.all_gather` returns
tensors detached from autograd for non-local ranks; we replace the local slot with
the in-graph tensor so backward propagates correctly.

For single-GPU use, gather is a no-op and this reduces to standard NT-Xent.
"""
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from .config import NT_XENT_TEMPERATURE


def _is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def _all_gather_with_grad(t: torch.Tensor) -> torch.Tensor:
    """all_gather across ranks while preserving gradient for local rank's slot."""
    if not _is_dist():
        return t
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    gathered = [torch.zeros_like(t) for _ in range(world_size)]
    dist.all_gather(gathered, t)
    gathered[rank] = t  # critical: keep local tensor in graph
    return torch.cat(gathered, dim=0)


class NTXentLoss(nn.Module):
    """SimCLR NT-Xent loss with DDP support.

    Args:
        temperature: Softmax temperature τ.
        gather_distributed: If True, gather z from all ranks before computing loss
            so anchors see negatives from all GPUs (true batch).
    """

    def __init__(
        self,
        temperature: float = NT_XENT_TEMPERATURE,
        gather_distributed: bool = True,
    ):
        super().__init__()
        self.temperature = temperature
        self.gather_distributed = gather_distributed

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """z1, z2: L2-normalized projections of shape (B, D)."""
        if self.gather_distributed and _is_dist():
            z1 = _all_gather_with_grad(z1)
            z2 = _all_gather_with_grad(z2)

        batch = z1.size(0)
        z = torch.cat([z1, z2], dim=0)  # (2B, D)

        sim = torch.matmul(z, z.t()) / self.temperature  # (2B, 2B)

        # Mask out self-similarity on the diagonal
        mask_self = torch.eye(2 * batch, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask_self, float("-inf"))

        # Positive pair targets: for i in [0..B), positive is i+B; for i in [B..2B), positive is i-B
        targets = torch.arange(2 * batch, device=z.device)
        targets = (targets + batch) % (2 * batch)

        loss = F.cross_entropy(sim, targets)
        return loss


def alignment_uniformity(z1: torch.Tensor, z2: torch.Tensor) -> dict:
    """Wang & Isola (2020) diagnostics. Collapse → uniformity ≈ 0."""
    with torch.no_grad():
        align = (z1 - z2).norm(dim=1).pow(2).mean().item()
        z = torch.cat([z1, z2], dim=0)
        pdist = torch.pdist(z, p=2).pow(2)
        uniform = pdist.mul(-2).exp().mean().log().item()
    return {"alignment": align, "uniformity": uniform}
