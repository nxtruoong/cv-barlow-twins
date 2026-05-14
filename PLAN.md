# Barlow Twins + ResNet-18 on State Farm Distracted Driver Detection

**Experimental study:** Evaluate Barlow Twins self-supervised pretraining as a solution for the State Farm Kaggle competition under free-tier resource constraints (Kaggle T4 x2, no paid compute). Method choice rationale and superseded SimCLR plan: see ADR-0003 and ADR-0001.

**Reproducibility seed:** `24521897` (apply everywhere — model init, GroupKFold, DataLoader workers, augmentation RNG).

---

## 1. Research Questions & Pre-Registered Success Criteria

Define before running experiments to avoid post-hoc rationalization.

| Hypothesis | Test | Pass | Marginal | Fail |
|---|---|---|---|---|
| **H1**: Barlow Twins pretrain improves over random init | B vs A val log loss | improvement ≥ 0.05 | 0.01 ≤ Δ < 0.05 | Δ < 0.01 |
| **H2**: Barlow Twins competitive with ImageNet pretrain | B vs C val log loss | B ≤ C + 0.10 | — | B − C > 0.10 |
| **H3**: Combined approach value | B + C ensemble vs C alone | ensemble > C | — | ensemble ≤ C |

Calibrate thresholds after Condition A fold 1 anchor result. Random baseline log loss ≈ 2.3.

---

## 2. Experimental Conditions

Three conditions, identical fine-tune protocol, only backbone init varies.

| Condition | Backbone init | Question answered |
|---|---|---|
| **A** Pure baseline | ResNet-18 random init | Supervised only on 22k labeled |
| **B** Barlow Twins (main) | ResNet-18 random + BT pretrain on 102k | Does BT help vs scratch? |
| **C** ImageNet | ResNet-18 ImageNet pretrained | Strong real-world baseline |

**Controlled variables (must match across A, B, C):** GroupKFold split, augmentation pipeline, optimizer, LR schedule, label smoothing, 2-stage fine-tune, TTA, K-fold ensemble, seed.

**Comparison protocol:** 1 fold per condition for headline numbers. 5-fold ensemble only on winning condition for Kaggle LB submission.

---

## 3. Data Engineering

### Dataset assembly
- **Pretrain (unlabeled)**: train (22k) ∪ test (~80k) = ~102k images. Transductive learning allowed by competition rules.
- **Fine-tune (labeled)**: 22k labeled train images only.
- **Resize**: 224×224 (after resize from native 640×480).

### Train/Val split — CRITICAL
Use `driver_imgs_list.csv` `subject` column. ~26 unique drivers.

```python
from sklearn.model_selection import GroupKFold
gkf = GroupKFold(n_splits=5)
folds = list(gkf.split(X, y, groups=subject_ids))
# Same fold ordering for A, B, C (deterministic via seed)
```

**Never random-split.** Random split causes massive val accuracy inflation due to subject leakage.

### Sanity checks (Day 1)
- Open 5 random train images, verify labels match folder
- Plot class distribution (confirm ~2.2k/class)
- Verify `driver_imgs_list.csv` filenames match image files
- Dry-run: 1 SimCLR epoch + 1 fine-tune epoch end-to-end

---

## 4. Barlow Twins Pretraining (Condition B only)

### Architecture
- **Backbone**: `models.resnet18(weights=None)` — random init.
- **Projector body**: `Linear(512→2048, bias=False) → BN → ReLU → Linear(2048→2048, bias=False) → BN → ReLU → Linear(2048→2048, bias=False)`.
- **Final per-view layer**: `BatchNorm1d(2048, affine=False)` — paper-mandated batch normalization that replaces manual `(z − mean) / std` before the cross-correlation matrix. Applied independently to each augmented view.
- **Loss**: Barlow Twins cross-correlation; λ_off = 5e-3.
- **All projector pieces discarded after Phase 1.**

### Augmentation pipeline
```python
import torchvision.transforms as T

pretrain_aug = T.Compose([
    T.RandomResizedCrop(224, scale=(0.5, 1.0)),  # NOT default 0.08
    T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    T.RandomGrayscale(p=0.2),
    T.RandomApply([T.GaussianBlur(23, sigma=(0.1, 2.0))], p=0.5),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# NO RandomHorizontalFlip — flips break left/right hand class identity
# Symmetric across views (not BYOL-style asymmetric); aug held constant
# across conditions A/B/C so SSL loss is the only variable.
```

### Multi-GPU + Loss Computation
**Setup**: 2× T4 via DDP, batch 256 (128 per GPU).

**Critical 1 — SyncBN on the final projector BN:**
```python
import torch.nn as nn
model = BarlowTwinsModel(...).cuda()
model = nn.SyncBatchNorm.convert_sync_batchnorm(model)  # BEFORE DDP wrap
model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
```
Without `SyncBN`, the final `BatchNorm1d(affine=False)` normalizes using per-GPU stats only → per-GPU C matrix → silent undertraining. Loss looks plausible, linear probe stalls.

**Critical 2 — global C matrix via cross-rank reduce:**
```python
# Inside BarlowTwinsLoss.forward (each rank):
c_local = z_a.T @ z_b               # local outer product, (D, D)
dist.all_reduce(c_local, op=dist.ReduceOp.SUM)
c = c_local / (local_n * world_size)  # divide by GLOBAL batch
# Loss computed identically on all ranks; DDP averages gradients normally.
```

**DO NOT use gradient accumulation** to fake larger batch. BT batch statistics in the final BN and C matrix would be computed on smaller-than-intended subsets per step.

### Optimizer & Schedule
```python
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
# 10-epoch linear warmup: 1e-6 → 1e-3
# Cosine decay epochs 10–80: 1e-3 → 0
from timm.scheduler import CosineLRScheduler
scheduler = CosineLRScheduler(
    optimizer, t_initial=80, warmup_t=10,
    warmup_lr_init=1e-6, lr_min=0,
)
```

Mixed precision: `torch.cuda.amp` (FP16) via `GradScaler`. LARS rejected — designed for batch ≥ 4096; at batch 256 its layer-wise trust ratio adds noise without benefit. Paper's wd=1.5e-6 dropped — keep 1e-4 to match Condition A optimizer (single-variable comparison); flag as tuning knob if linear probe stalls.

### Diagnostics during pretrain — REQUIRED

**1. C-matrix statistics — every epoch (free, computed during loss):**
```python
diag_mean = torch.diagonal(C).mean()        # target → 1.0
off_rms   = (C[~eye(D, bool)]**2).mean().sqrt()  # target → 0.05–0.1
# Collapse signature: diag_mean stuck below 0.3 by epoch 20.
```

**2. Linear probe accuracy — at epochs 20, 40, 60, 80 only** (heavier; gating signal already covered by C-matrix stats):
```python
# Freeze backbone, train Linear(512 → 10) on 1 fold of labeled data
# 50 epochs, AdamW lr=1e-3
# Healthy: ep20 ~30–40%, ep40 ~55%, ep60 ~70%, ep80 ~75%
# Stuck near random (10%) by ep20 → SSL broken; pre-registered abort.
```

**3. Pre-registered failure fallback** (per ADR-0003):
At epoch 20, if ALL of {linear probe < 25%, c_diag_mean < 0.3, off_rms > 0.4}:
- Do NOT abort training (sunk cost — let final ckpt complete and report)
- Start Condition A (scratch) fine-tune in parallel
- Report BT honestly as "collapsed under budget" + working A vs C comparison

### Checkpointing
- Save every 10 epochs to `/kaggle/working/bt/`
- Filename: `bt_resnet18_ep{N}.pth` + sliding `bt_resnet18_latest.pth`
- Upload `_latest.pth` to a Kaggle Dataset between sessions for resume

### Time budget
- ~400 iter/epoch (102k / 256) × ~1.0s/iter ≈ 7 min/epoch on T4 x2 (SyncBN ~5% overhead, 2048³ projector, fp16)
- 80 epochs ≈ 9.3 hours of pure training + diagnostics + Kaggle session boot
- Split across 2 sessions (5h × 40 epochs each), resume via `_latest.pth`

---

## 5. Fine-Tuning Protocol (all 3 conditions)

### Augmentation (lighter than pretrain)
```python
finetune_aug_train = T.Compose([
    T.RandomResizedCrop(224, scale=(0.7, 1.0)),
    T.ColorJitter(0.2, 0.2, 0.2),
    T.RandomApply([T.GaussianBlur(5)], p=0.3),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
finetune_aug_val = T.Compose([
    T.Resize(256), T.CenterCrop(224), T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# NO horizontal flip (class identity)
```

### Architecture change
- Load pretrained backbone (or scratch / ImageNet, per condition)
- Drop SimCLR projection head if applicable
- Add `Linear(512 → 10)` classifier head

### Stage 1 — Freeze backbone, train classifier (5 epochs)
```python
for p in backbone.parameters(): p.requires_grad = False
optim = AdamW(classifier.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = CrossEntropyLoss(label_smoothing=0.1)
```

### Stage 2 — Unfreeze all, discriminative LR (25 epochs)
```python
for p in model.parameters(): p.requires_grad = True

param_groups = [
    {'params': backbone.conv1.parameters(),  'lr': 1e-5},
    {'params': backbone.layer1.parameters(), 'lr': 1e-5},
    {'params': backbone.layer2.parameters(), 'lr': 5e-5},
    {'params': backbone.layer3.parameters(), 'lr': 5e-5},
    {'params': backbone.layer4.parameters(), 'lr': 1e-4},
    {'params': classifier.parameters(),      'lr': 1e-3},
]
optim = AdamW(param_groups, weight_decay=1e-4)
scheduler = CosineLRScheduler(optim, t_initial=25, warmup_t=2)

# Early stop: monitor val log loss, patience=5
```

### Model selection
Pick checkpoint with **lowest val log loss**, not best accuracy. (Competition metric is log loss; accuracy can mislead due to overconfident predictions.)

---

## 6. Submission Strategy (winner condition only)

### TTA — 5 crops, no flip
```python
# Generate 5 augmented views (center + 4 corners), average softmax probs
# DO NOT use horizontal flip — breaks class identity
```

### 5-fold ensemble
```python
# Train winner condition on all 5 GroupKFold splits
# Average probs across all 5 fold models for final submission
```

### Probability clipping
```python
probs = np.clip(probs, 1e-15, 1 - 1e-15)  # required to bound log loss
```

### Submission file format
```python
# sample_submission.csv columns: img, c0, c1, ..., c9
submission = pd.DataFrame({'img': test_filenames})
for i in range(10):
    submission[f'c{i}'] = probs[:, i]
submission.to_csv('submission.csv', index=False)
```

### Save intermediates
Save raw fold probabilities as `.npy` after each fold. Enables re-ensembling without retraining.

---

## 7. Demo Pipeline (for teacher presentation)

### Saved artifacts (end of fine-tune, per condition)
```python
torch.save({
    'model_state_dict': model.state_dict(),
    'class_names': [
        'safe driving', 'texting - right', 'phone - right',
        'texting - left', 'phone - left', 'operating radio',
        'drinking', 'reaching behind', 'hair/makeup',
        'talking to passenger',
    ],
    'preprocessing': {'resize': 224, 'mean': [0.485, 0.456, 0.406],
                      'std': [0.229, 0.224, 0.225]},
    'condition': 'B_bt',  # or 'A_scratch' / 'C_imagenet'
}, f'demo_bundle_{condition}.pth')
```

### Inference + Grad-CAM (`demo.ipynb`)
```python
import torch, torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt
import numpy as np

transform = T.Compose([
    T.Resize(256), T.CenterCrop(224), T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict(img_path, model, tta=True):
    img = Image.open(img_path).convert('RGB')
    if tta:
        views = [transform(img).unsqueeze(0).cuda() for _ in range(5)]
        probs = torch.stack([F.softmax(model(v), dim=1) for v in views]).mean(0)
    else:
        probs = F.softmax(model(transform(img).unsqueeze(0).cuda()), dim=1)
    return probs.squeeze().detach().cpu().numpy()

def show_with_gradcam(img_path, model, class_names, target_layer):
    probs = predict(img_path, model)
    top3 = probs.argsort()[-3:][::-1]

    img = np.array(Image.open(img_path).convert('RGB').resize((224, 224))) / 255.0
    img_tensor = transform(Image.open(img_path).convert('RGB')).unsqueeze(0).cuda()

    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=img_tensor)[0]
    cam_vis = show_cam_on_image(img.astype(np.float32), grayscale_cam, use_rgb=True)

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    ax[0].imshow(img); ax[0].axis('off')
    ax[0].set_title(f'Pred: {class_names[top3[0]]} ({probs[top3[0]]:.1%})')
    ax[1].barh(range(10), probs)
    ax[1].set_yticks(range(10)); ax[1].set_yticklabels(class_names)
    ax[1].set_xlabel('Probability'); ax[1].set_xlim(0, 1)
    ax[2].imshow(cam_vis); ax[2].axis('off')
    ax[2].set_title('Grad-CAM')
    plt.tight_layout(); plt.show()
```

### Demo image selection
- Google search: "driver texting in car dashcam", "driver drinking coffee dashboard"
- Pick 5-8 images covering 5+ classes
- Include 1 deliberately hard case (sunglasses, different cabin) to honestly show failure mode
- Avoid: cartoon, top-down, exterior shots

### Talking points for teacher
- **Domain gap is expected.** State Farm is narrow distribution; Google images are OOD. Degraded confidence is correct behavior, not a bug.
- **Compare 3 conditions side-by-side on same image.** Expected: ImageNet (C) generalizes best, Barlow Twins (B) most overfit to State Farm cabin specifics.
- **Grad-CAM proves the model isn't memorizing noise.** For texting class, heatmap should highlight hands/phone region. If it focuses on face only → spurious features → discussion point.

---

## 8. Reproducibility

```python
import torch, numpy as np, random, os

SEED = 24521897

def set_seed(seed=SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # slower but reproducible

set_seed()
# Also pass seed=SEED to GroupKFold, DataLoader generator, etc.
```

---

## 9. Timeline

| Day | Task | Hours |
|---|---|---|
| 1 | Pipeline code + augmentation + dry run (1 epoch each stage) | 4 |
| 2 | Barlow Twins pretrain session 1 (epoch 1–40) + linear probe at ep 20/40 | 5 |
| 3 | Barlow Twins pretrain session 2 (epoch 41–80) + linear probe at ep 60/80 | 5 |
| 4 AM | Fine-tune all 3 conditions (1 fold each) | 4 |
| 4 PM | Pick winner, 5-fold ensemble on winner, LB submission | 4 |
| 5 | Save 3 demo bundles, build `demo.ipynb` with Grad-CAM, pick Google images, screenshots for slides | 3 |

---

## 10. Reporting Template

Fill after experiments complete.

### Headline table
| Condition | Val log loss (fold 1) | LB log loss (single) | LB log loss (5-fold ensemble) |
|---|---|---|---|
| A scratch | ? | ? | (winner only) |
| B Barlow Twins | ? | ? | (winner only) |
| C ImageNet | ? | ? | (winner only) |

### Verdict
- H1: [ ] Pass [ ] Marginal [ ] Fail
- H2: [ ] Pass [ ] Marginal [ ] Fail
- H3: [ ] Pass [ ] Fail

### Linear probe progression (Condition B)
| Epoch | Linear probe acc | BT loss | C diag mean | Off-diag rms |
|---|---|---|---|---|
| 20 | ? | ? | ? | ? |
| 40 | ? | ? | ? | ? |
| 60 | ? | ? | ? | ? |
| 80 | ? | ? | ? | ? |

### Discussion points
- Did Barlow Twins features transfer to OOD Google images better/worse than ImageNet?
- What did Grad-CAM reveal about feature locality?
- Was 80 epochs enough or did probe still improve at the end?
- Did C-matrix diag/off-diag converge to ~1.0 / ~0.05, or did either stall?

---

## Appendix A — Methods Not Chosen (with trade-offs)

| Method | Why not chosen | When to revisit |
|---|---|---|
| ImageNet pretrained + supervised only (no SSL) | Defeats experimental purpose | If only goal is LB rank |
| **SimCLR + NT-Xent** | Popular Kaggle default → unoriginal; requires large batch + differentiable cross-rank all-gather plumbing. Superseded per ADR-0003; code retained on `tpu-v5e-8-snapshot` branch | If BT collapses and a known-working fallback is needed |
| MoCo v2 | Queue-based negatives, memory-efficient. Trade-off: more moving parts than BT (queue, momentum encoder); also contrastive-family | If batch 256 still insufficient for BT |
| BYOL / SimSiam | No negatives, smaller batch OK. Trade-off: collapse risk without careful momentum / predictor design; same originality slot as BT but more failure modes | If BT collapses despite diagnostics |
| MAE / SparK (masked image modeling on ConvNets) | Originality high but sparse-conv kernel deps unproven on Kaggle; schedule risk | Post-deadline experiment |
| DINO with ResNet | Self-distillation, no negatives. Trade-off: momentum encoder + multi-crop ~2.5× compute over budget on T4 | If a future budget allows ~25hr pretrain |
| Rotation pretext (Gidaris 2018) | Textbook SSL; teacher would call it more unoriginal than SimCLR | Never (in this project) |
| SupCon (Supervised Contrastive) | Higher accuracy. Trade-off: not self-supervised, ignores unlabeled test data | If H2 fails and labeled-data SOTA is the goal |
| Pseudo-labeling | Train supervised, label test set, retrain. Trade-off: noisy labels amplify errors | Day 5 stretch goal if time permits |
| LARS optimizer | Designed for batch ≥ 4096. Layer-wise trust ratio adds noise at batch 256 | If scaling to multi-node compute |
| Gradient accumulation for batch >256 | Breaks BT batch statistics (final BN and C matrix computed on smaller-than-intended subsets per step) | Never for BT |
| RandomHorizontalFlip in augmentation | Flips left/right hand → destroys class identity (texting-left vs texting-right) | Never on this dataset |
| Random train/val split | Subject leakage massively inflates val acc vs LB | Never on this dataset |
