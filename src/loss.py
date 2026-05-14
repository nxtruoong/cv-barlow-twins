"""NT-Xent loss with differentiable cross-rank all-gather.

Two backends, picked at runtime:

1. **PyTorch/XLA (TPU)**: `xm.all_gather` is non-differentiable by default.
   We wrap it in a custom autograd.Function whose backward does an
   `xm.reduce_scatter("sum", grad, ...)` to route the per-rank slice of
   the gradient back to the originating rank.

2. **torch.distributed (CUDA, gloo/nccl)**: classic pattern — call
   `dist.all_gather` then overwrite the local slot with the in-graph
   tensor so backward still flows for that rank's contribution.

For single-device runs both paths reduce to identity, giving standard NT-Xent.
"""
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from .config import NT_XENT_TEMPERATURE


try:
    import torch_xla.core.xla_model as xm  # type: ignore
    _HAS_XLA = True
except ImportError:
    xm = None
    _HAS_XLA = False


def _is_torch_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def _is_xla_dist() -> bool:
    if not _HAS_XLA:
        return False
    try:
        return xm.xrt_world_size() > 1
    except Exception:
        return False


class _DifferentiableAllGatherXLA(torch.autograd.Function):
    """xm.all_gather forward; reduce_scatter("sum", ...) backward.

    Why: naive xm.all_gather is a no-grad collective. Without this wrapper,
    NT-Xent gradients on remote-rank slices are silently zero -> effective
    batch collapses to per-core size -> SimCLR signal degrades.
    """

    @staticmethod
    def forward(ctx, t: torch.Tensor) -> torch.Tensor:
        ctx.world_size = xm.xrt_world_size()
        ctx.rank = xm.get_ordinal()
        ctx.local_shape = t.shape
        return xm.all_gather(t, dim=0)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # reduce_scatter sums grads from all ranks then splits along dim 0;
        # rank r receives the slice that corresponds to its forward input.
        grad_in = xm.reduce_scatter(
            xm.REDUCE_SUM, grad_output, scale=1.0, scatter_dim=0,
            shard_count=ctx.world_size,
        )
        return grad_in


def _all_gather_xla(t: torch.Tensor) -> torch.Tensor:
    return _DifferentiableAllGatherXLA.apply(t)


def _all_gather_torch_dist(t: torch.Tensor) -> torch.Tensor:
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    gathered = [torch.zeros_like(t) for _ in range(world_size)]
    dist.all_gather(gathered, t)
    gathered[rank] = t  # critical: keep local tensor in graph
    return torch.cat(gathered, dim=0)


def _all_gather_with_grad(t: torch.Tensor) -> torch.Tensor:
    """Dispatch to XLA or torch.distributed; identity if not distributed."""
    if _is_xla_dist():
        return _all_gather_xla(t)
    if _is_torch_dist():
        return _all_gather_torch_dist(t)
    return t


class NTXentLoss(nn.Module):
    """SimCLR NT-Xent loss with XLA + DDP support.

    Args:
        temperature: Softmax temperature τ.
        gather_distributed: If True, gather z from all ranks/cores before
            computing loss so anchors see negatives from the full global batch.
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
        if self.gather_distributed:
            z1 = _all_gather_with_grad(z1)
            z2 = _all_gather_with_grad(z2)

        batch = z1.size(0)
        z = torch.cat([z1, z2], dim=0)  # (2B, D)

        sim = torch.matmul(z, z.t()) / self.temperature  # (2B, 2B)

        # Mask out self-similarity on the diagonal.
        mask_self = torch.eye(2 * batch, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(mask_self, float("-inf"))

        # Positive pair targets: i in [0..B) -> i+B; i in [B..2B) -> i-B.
        targets = torch.arange(2 * batch, device=z.device)
        targets = (targets + batch) % (2 * batch)

        return F.cross_entropy(sim, targets)


def alignment_uniformity(z1: torch.Tensor, z2: torch.Tensor) -> dict:
    """Wang & Isola (2020) diagnostics. Collapse → uniformity ≈ 0."""
    with torch.no_grad():
        align = (z1 - z2).norm(dim=1).pow(2).mean().item()
        z = torch.cat([z1, z2], dim=0)
        pdist = torch.pdist(z, p=2).pow(2)
        uniform = pdist.mul(-2).exp().mean().log().item()
    return {"alignment": align, "uniformity": uniform}
