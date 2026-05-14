"""ResNet-18 backbone + SimCLR projection head + supervised classifier."""
from typing import Optional, Dict
import torch
import torch.nn as nn
from torchvision import models

from .config import (
    NUM_CLASSES, PROJECTION_DIM, PROJECTION_HIDDEN,
    BT_PROJECTION_DIM, BT_PROJECTION_HIDDEN,
)


def _build_backbone(pretrained: bool = False) -> nn.Module:
    """ResNet-18 with fc replaced by Identity. Output dim = 512."""
    if pretrained:
        net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    else:
        net = models.resnet18(weights=None)
    feature_dim = net.fc.in_features  # 512
    net.fc = nn.Identity()
    net.feature_dim = feature_dim
    return net


class ProjectionHead(nn.Module):
    """MLP head for SimCLR: Linear -> ReLU -> Linear."""

    def __init__(self, in_dim: int = 512, hidden_dim: int = PROJECTION_HIDDEN,
                 out_dim: int = PROJECTION_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BarlowTwinsProjector(nn.Module):
    """Barlow Twins projector body (no final BN).

    Linear -> BN -> ReLU -> Linear -> BN -> ReLU -> Linear.

    The final per-view ``BatchNorm1d(affine=False)`` lives on the parent
    ``BarlowTwinsModel`` so it can be applied to each augmented view
    independently (paper section 3, official ref impl).

    Linears use ``bias=False`` because BN absorbs the shift. No ReLU after
    the final linear: BT requires signed features for the cross-correlation
    matrix to span [-1, 1].
    """

    def __init__(
        self,
        in_dim: int = 512,
        hidden_dim: int = BT_PROJECTION_HIDDEN,
        out_dim: int = BT_PROJECTION_DIM,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BarlowTwinsModel(nn.Module):
    """Backbone + Barlow Twins projector + final per-view BN(affine=False).

    Two forward modes:

    * ``model(x)`` — single view, returns BN-normalized z. Used by linear
      probe and downstream inference.
    * ``model(v1, v2)`` — augmented pair, returns ``(z1, z2)``. Uses a
      single concatenated forward through backbone + projector (cheaper +
      avoids DDP "multiple-forward" autograd quirks) then applies the
      final BN separately per view to match the BT paper.

    Under DDP the final BN MUST be converted to SyncBatchNorm so the C
    matrix reflects the global batch. ``BarlowTwinsModel.convert_sync_bn(m)``
    is a convenience that calls ``nn.SyncBatchNorm.convert_sync_batchnorm``.
    """

    def __init__(self, pretrained_backbone: bool = False):
        super().__init__()
        self.backbone = _build_backbone(pretrained=pretrained_backbone)
        self.projector = BarlowTwinsProjector(in_dim=self.backbone.feature_dim)
        self.final_bn = nn.BatchNorm1d(BT_PROJECTION_DIM, affine=False)

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(self.backbone(x))

    def forward(self, x: torch.Tensor, second: Optional[torch.Tensor] = None):
        if second is None:
            y = self._embed(x)
            return self.final_bn(y)
        v = torch.cat([x, second], dim=0)
        y = self._embed(v)
        y1, y2 = y.chunk(2, dim=0)
        return self.final_bn(y1), self.final_bn(y2)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    @staticmethod
    def convert_sync_bn(model: nn.Module) -> nn.Module:
        return nn.SyncBatchNorm.convert_sync_batchnorm(model)


class SimCLRModel(nn.Module):
    """Backbone + projection head. Returns L2-normalized projection."""

    def __init__(self, pretrained_backbone: bool = False):
        super().__init__()
        self.backbone = _build_backbone(pretrained=pretrained_backbone)
        self.projection = ProjectionHead(in_dim=self.backbone.feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        z = self.projection(h)
        return nn.functional.normalize(z, dim=1)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Backbone features only (no projection). For linear probe / fine-tune."""
        return self.backbone(x)


class ClassifierModel(nn.Module):
    """Backbone + Linear classifier head. Used for all 3 fine-tune conditions."""

    def __init__(self, pretrained_backbone: bool = False,
                 num_classes: int = NUM_CLASSES):
        super().__init__()
        self.backbone = _build_backbone(pretrained=pretrained_backbone)
        self.classifier = nn.Linear(self.backbone.feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        return self.classifier(h)


def load_ssl_backbone_into_classifier(
    classifier: ClassifierModel, ssl_state_dict: Dict[str, torch.Tensor]
) -> None:
    """Copy SSL backbone weights (SimCLR or Barlow Twins) into classifier.

    Filters by ``backbone.`` prefix and discards the projection head — works
    for either SSL method since both share the ResNet-18 backbone layout.
    """
    backbone_state = {
        k.replace("backbone.", "", 1): v
        for k, v in ssl_state_dict.items()
        if k.startswith("backbone.")
    }
    missing, unexpected = classifier.backbone.load_state_dict(backbone_state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys when loading SSL backbone: {unexpected}")
    return missing, unexpected


# Back-compat alias for older imports.
load_simclr_backbone_into_classifier = load_ssl_backbone_into_classifier


def freeze_backbone(model: ClassifierModel) -> None:
    for p in model.backbone.parameters():
        p.requires_grad = False


def unfreeze_backbone(model: ClassifierModel) -> None:
    for p in model.backbone.parameters():
        p.requires_grad = True


def discriminative_param_groups(model: ClassifierModel) -> list:
    """Per-layer LR for stage-2 fine-tune."""
    return [
        {"params": model.backbone.conv1.parameters(), "lr": 1e-5},
        {"params": model.backbone.bn1.parameters(), "lr": 1e-5},
        {"params": model.backbone.layer1.parameters(), "lr": 1e-5},
        {"params": model.backbone.layer2.parameters(), "lr": 5e-5},
        {"params": model.backbone.layer3.parameters(), "lr": 5e-5},
        {"params": model.backbone.layer4.parameters(), "lr": 1e-4},
        {"params": model.classifier.parameters(), "lr": 1e-3},
    ]


def build_classifier_for_condition(
    condition: str,
    ssl_ckpt_path: Optional[str] = None,
    simclr_ckpt_path: Optional[str] = None,  # back-compat
) -> ClassifierModel:
    """Build classifier for one of {A_scratch, B_bt, B_simclr, C_imagenet}.

    ``B_bt`` is the Phase 1 Barlow Twins condition (current default per ADR-0003).
    ``B_simclr`` retained for fallback / ablation against the prior plan.
    Both load the backbone via ``load_ssl_backbone_into_classifier`` —
    checkpoint format is identical.
    """
    if condition == "A_scratch":
        return ClassifierModel(pretrained_backbone=False)
    if condition == "C_imagenet":
        return ClassifierModel(pretrained_backbone=True)
    if condition in ("B_bt", "B_simclr"):
        ckpt_path = ssl_ckpt_path or simclr_ckpt_path
        if ckpt_path is None:
            raise ValueError(f"{condition} requires ssl_ckpt_path")
        model = ClassifierModel(pretrained_backbone=False)
        state = torch.load(ckpt_path, map_location="cpu")
        sd = state.get("model_state_dict", state)
        load_ssl_backbone_into_classifier(model, sd)
        return model
    raise ValueError(f"Unknown condition: {condition}")
