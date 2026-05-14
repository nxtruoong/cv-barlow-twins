# Kaggle Quickref — Barlow Twins Pretrain

Paste-ready cells for Kaggle T4 x2. See [RUNBOOK.md](RUNBOOK.md) for full procedure and [PLAN.md](PLAN.md) for design rationale.

---

## Session settings

- **Accelerator:** GPU T4 x2
- **Internet:** On
- **Persistence:** off
- **Input:** `state-farm-distracted-driver-detection` (Add Competition)

Time budget per session: ~5h. Two sessions × 40 epochs each = 80 ep total.

---

## Session 1 — epochs 1–40 (~4.5h)

### Cell 1 — clone + install

```python
!git clone https://github.com/nxtruoong/cv-barlow-twins.git /kaggle/working/CV
!pip install -q timm
```

### Cell 2 — verify data + GPUs

```python
!ls /kaggle/input/state-farm-distracted-driver-detection/imgs/train/ | head
!nvidia-smi -L
```

Expect 10 class dirs `c0..c9` + 2 Tesla T4 GPUs.

### Cell 3 — launch pretrain

```python
exec(open("/kaggle/working/CV/notebooks/kaggle_pretrain.py").read())
```

`RESUME = None` already set in the script for session 1.

### Cell 4 — verify checkpoint, prep for session 2

```python
!ls -la /kaggle/working/bt/
```

Expect `bt_resnet18_ep039.pth`, `bt_resnet18_latest.pth`, `history.json`.

**Download `bt_resnet18_latest.pth` to local disk** before closing notebook (`/kaggle/working` wipes between sessions).

### Between sessions — upload checkpoint

1. https://www.kaggle.com/datasets → **New Dataset**
2. Title: `bt-ckpt-resume`
3. Upload the `.pth` file
4. Set visibility (Private OK)

---

## Session 2 — epochs 41–80 (~4.5h)

### Setup

- Same notebook OR new notebook
- Same accelerator (GPU T4 x2) + Internet On
- **Add Input** → search `bt-ckpt-resume` (your uploaded dataset)
- **Add Input** → `state-farm-distracted-driver-detection` (Competition)

### Cell 1 — clone + install (same as session 1)

```python
!git clone https://github.com/nxtruoong/cv-barlow-twins.git /kaggle/working/CV
!pip install -q timm
```

### Cell 2 — launch with resume (inline, no file edit)

```python
import sys, os, argparse
REPO = "/kaggle/working/CV"
sys.path.insert(0, REPO)
os.environ["PYTHONPATH"] = REPO + os.pathsep + os.environ.get("PYTHONPATH", "")

import torch, torch.multiprocessing as mp
from src.pretrain import run_pretrain, ddp_worker
from src.config import BT_BATCH_SIZE, BT_EPOCHS, BT_LR, BT_LAMBDA

args = argparse.Namespace(
    epochs=BT_EPOCHS,
    batch_size=BT_BATCH_SIZE,
    lr=BT_LR,
    lambda_off=BT_LAMBDA,
    num_workers=2,
    amp=True,
    save_every=10,
    resume="/kaggle/input/bt-ckpt-resume/bt_resnet18_latest.pth",
    output_dir="/kaggle/working/bt",
)
world_size = torch.cuda.device_count()
if world_size > 1:
    mp.spawn(ddp_worker, args=(world_size, args), nprocs=world_size, join=True)
else:
    run_pretrain(args)
```

First print should be:
```
Resumed from /kaggle/input/bt-ckpt-resume/bt_resnet18_latest.pth at epoch 40
```

### Cell 3 — final checkpoint

```python
!ls -la /kaggle/working/bt/
```

Expect `bt_resnet18_ep079.pth`. **Download it.** Upload as Kaggle Dataset `bt-pretrain-final` for the Day 4 finetune step.

---

## Healthy training trajectory

Watch the tqdm postfix per step (sampled every 20 iters):

```
loss=XXXX  diag=0.YY  off=0.ZZ
```

| Epoch | Loss | `c_diag_mean` | `c_offdiag_rms` | Linear probe |
|---|---|---|---|---|
| 0 | ~2000 | ~0.05 | ~0.4 | — |
| 10 | ~500 | ~0.4 | ~0.2 | — |
| 20 | ~200 | ~0.6 | ~0.12 | ~0.35 |
| 40 | ~80 | ~0.8 | ~0.08 | ~0.55 |
| 80 | ~30 | ~0.95 | ~0.05 | ~0.75 |

**Abort criteria at epoch 20** (per ADR-0003): if `probe_acc < 0.25` AND `c_diag_mean < 0.3` AND `c_offdiag_rms > 0.4` → BT collapsed; start scratch (Condition A) finetune in parallel, let BT finish, report outcome honestly.

---

## Common Kaggle gotchas

| Symptom | Cause | Fix |
|---|---|---|
| `gh: not found` | No GitHub CLI on Kaggle | Use `git clone` directly (already in cells) |
| `ModuleNotFoundError: timm` | First-run dep missing | Cell 1 installs it; rerun if you skipped |
| `RuntimeError: ... mp.spawn` after restart | Stale CUDA ctx from prior run | **Restart notebook** (top bar → Restart & Clear) |
| `c_diag_mean` stuck near 0 by ep 5 | SyncBN not applied | Check `src/pretrain.py` calls `BarlowTwinsModel.convert_sync_bn` before DDP. Re-clone repo. |
| `it/s` < 0.5 | Data pipeline bottleneck | Bump `num_workers=4` in args (Kaggle has 4 vCPU shared between 2 ranks) |
| `CUDA out of memory` | Aug + 2 views + 2048³ projector tight on T4 16GB | Drop `batch_size` from 256 → 192. Loss tolerates it. |
| `/kaggle/working` empty after relog | Session expired, working dir wiped | Always download `.pth` BEFORE closing. Reattach as Dataset input next session. |
| `Save Version` produces empty output | Save runs fresh kernel, not interactive state | Use download-then-reupload pattern; do not rely on Save Version for ckpts |

---

## After pretrain: finetune (Day 4)

Same pattern — see `notebooks/kaggle_finetune.py`:

```python
!git clone https://github.com/nxtruoong/cv-barlow-twins.git /kaggle/working/CV
!pip install -q timm
exec(open("/kaggle/working/CV/notebooks/kaggle_finetune.py").read())
```

Requires `bt-pretrain-final` dataset attached. Trains A_scratch, B_bt, C_imagenet on fold 0; prints headline log losses.
