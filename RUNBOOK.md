# Project Runbook — SimCLR + ResNet-18 on State Farm

End-to-end execution guide. Pair with `PLAN.md` (experimental design) and `README.md` (overview).

Estimated wall clock: **3-4 days**, ~15h Kaggle GPU budget.

Pretrain finishes in ~8-9h on T4 x2 (200 epochs × 2.3 min/epoch + diagnostics).
Target 1 session if you can babysit; 2 sessions if not.

---

## Day 0 — Local prep (one-time, ~15 min)

### 0.1 Verify repo state
```bash
git status                    # clean
git log --oneline -3          # see latest commits
```

### 0.2 Kaggle account requirements
- Verified phone number (required for GPU + internet in notebooks)
- Joined State Farm competition: https://www.kaggle.com/c/state-farm-distracted-driver-detection
- Accepted competition rules

### 0.3 Push repo (if not already)
```bash
git push origin main
```

---

## Day 1 — Sanity check (~30 min, CPU fine)

**Goal:** verify data loads, GroupKFold has no leakage, files exist. Cheap insurance before burning GPU days.

### 1.1 Create Kaggle notebook
1. https://www.kaggle.com/code → **New Notebook**
2. Right sidebar → **Add Input** → search "State Farm Distracted Driver Detection" → **Add**
3. Settings:
   - Accelerator: **None** (CPU suffices for sanity)
   - Internet: **On** (for `git clone`)
   - Persistence: **Files only**

### 1.2 Clone repo
Cell 1:
```python
!git clone https://github.com/nxtruoong/CV.git /kaggle/working/CV
!pip install -q timm
```

### 1.3 Run sanity check
Cell 2:
```python
exec(open("/kaggle/working/CV/notebooks/kaggle_sanity_check.py").read())
```

### 1.4 Expected output
```
Labeled images: 22424
Unique drivers: 26
Test images: 79726

Class distribution:
c0    2489
c1    2267
...

Fold 0: train ... overlap=0
Fold 1: train ... overlap=0
...
All fold subject overlaps = 0. GroupKFold OK.

[2x5 grid of class samples renders]

--- Sanity check complete. Ready for pretrain. ---
```

### 1.5 Failure modes
| Symptom | Cause | Fix |
|---|---|---|
| `FileNotFoundError driver_imgs_list.csv` | Dataset not attached or wrong path | Check `!ls /kaggle/input/competitions/`; update `src/config.py:KAGGLE_INPUT` if slug differs |
| `Subject leakage in fold N` assert | Bug in GroupKFold | Stop, do not proceed |
| `ModuleNotFoundError: timm` | Install cell skipped | Re-run cell 1 |
| Image count != 22424 | Partial dataset | Re-attach competition input |

---

## Day 2 — SimCLR pretrain (1 session if possible, ~8-9h)

**Goal:** train SimCLR end-to-end in single session. Fallback: 2 sessions.

**Timing math:**
- 399 steps/epoch × ~350ms/step ≈ 2.3 min/epoch
- 200 epochs × 2.3 min ≈ 7.7h training + ~20 min diagnostics
- Kaggle interactive cap ~9h → tight but feasible

**Mid-run sanity:** after epoch 1 stabilizes, check tqdm `it/s`:
- ≥ 3 it/s → on track for 1 session
- < 2 it/s → data loader bottleneck. Stop, bump num_workers=8, restart
- ETA = (399 / it_per_sec × 200) / 3600 hours

### 2.1 New notebook
Same data input attached. Settings:
- Accelerator: **GPU T4 x2**
- Internet: **On**

### 2.2 Clone + install
```python
!git clone https://github.com/nxtruoong/CV.git /kaggle/working/CV
!pip install -q timm
```

### 2.3 Launch pretrain
```python
exec(open("/kaggle/working/CV/notebooks/kaggle_pretrain.py").read())
```

`RESUME = None` already set. DDP via `mp.spawn` auto-detects 2 GPUs.

### 2.4 Monitor
- tqdm shows per-epoch loss
- Every 10 epochs: `align`, `uniform`, `probe_acc`, `probe_ll` printed
- Checkpoint saved every 10 epochs to `/kaggle/working/simclr/`

**Healthy trajectory:**
| Epoch | Loss | probe_acc |
|---|---|---|
| 0 | ~6.5 | ~0.10 |
| 10 | ~5.5 | ~0.20 |
| 30 | ~4.5 | ~0.45 |
| 80 | ~4.0 | ~0.65 |

If `probe_acc` < 0.20 at epoch 30 → **STOP**. Aug or temperature broken. Debug.

### 2.5 If session finishes (200 epochs done)
1. Verify final ckpt:
   ```python
   !ls -la /kaggle/working/simclr/
   ```
   Expect `simclr_resnet18_ep199.pth` + `simclr_resnet18_latest.pth` + `history.json`.
2. Download `simclr_resnet18_ep199.pth` + `history.json`
3. Upload as Kaggle Dataset titled `simclr-pretrain-final`
4. Skip Day 3, jump to Day 4 PM (finetune)

### 2.6 If session times out before ep200 (fallback)
1. Note epoch reached (e.g. ep~150)
2. Download `simclr_resnet18_latest.pth`
3. Upload as Kaggle Dataset `simclr-ckpt-resume`
4. Go to Day 3 (resume session)

---

## Day 3 — Pretrain resume session (only if Day 2 timed out)

### 3.1 New notebook
- GPU T4 x2, Internet On
- Attach inputs:
  - State Farm Distracted Driver Detection (competition)
  - `simclr-ckpt-resume` (Day 2 partial ckpt)

### 3.2 Clone + set resume path
```python
!git clone https://github.com/nxtruoong/CV.git /kaggle/working/CV
!pip install -q timm
```

Edit `notebooks/kaggle_pretrain.py` line ~32 before exec, OR inline:
```python
import sys, os, argparse
REPO = "/kaggle/working/CV"
sys.path.insert(0, REPO)
os.environ["PYTHONPATH"] = REPO + os.pathsep + os.environ.get("PYTHONPATH", "")

import torch, torch.multiprocessing as mp
from src.pretrain import run_pretrain, ddp_worker

RESUME = "/kaggle/input/simclr-ckpt-resume/simclr_resnet18_latest.pth"
args = argparse.Namespace(
    epochs=200, batch_size=256, lr=1e-3, num_workers=4, amp=True,
    save_every=10, diagnostic_every=10, resume=RESUME,
    output_dir="/kaggle/working/simclr",
)
world_size = torch.cuda.device_count()
if world_size > 1:
    mp.spawn(ddp_worker, args=(world_size, args), nprocs=world_size, join=True)
else:
    run_pretrain(args)
```

### 3.3 Verify resume
First print after launch:
```
Resumed from /kaggle/input/simclr-ckpt-resume/simclr_resnet18_latest.pth at epoch <N>
```

### 3.4 End of session
Download `simclr_resnet18_ep199.pth` + `history.json`. Upload as `simclr-pretrain-final`.

---

## Day 4 PM — Fine-tune 3 conditions, fold 0 (~3h)

**Goal:** A_scratch vs B_simclr vs C_imagenet on fold 0. Pick winner by val log loss.

### 4.3 New notebook
- GPU T4 x2 (only 1 used, but parallel sessions allowed)
- Attach: competition data + `simclr-pretrain-final`

### 4.4 Run finetune
```python
!git clone https://github.com/nxtruoong/CV.git /kaggle/working/CV
!pip install -q timm
exec(open("/kaggle/working/CV/notebooks/kaggle_finetune.py").read())
```

Trains 3 models sequentially. ~45 min each.

### 4.5 Read headline output
```
=========== Headline ===========
  A_scratch:  val_log_loss = 0.XXXX
  B_simclr:   val_log_loss = 0.XXXX
  C_imagenet: val_log_loss = 0.XXXX

Winner: <condition>
```

Record numbers in `PLAN.md` § 10 table.

### 4.6 Hypothesis check
- **H1 (SSL > scratch):** `B - A ≥ 0.05`?
- **H2 (SSL ≈ ImageNet):** `B - C ≤ 0.10`?

Save all 3 bundles (download from `/kaggle/working/finetune/`).

---

## Day 4 Late — 5-fold ensemble of winner (~4h)

### 4.7 New notebook, same inputs
```python
!git clone https://github.com/nxtruoong/CV.git /kaggle/working/CV
!pip install -q timm
```

Edit `kaggle_finetune_kfold.py` line 15 (or inline):
```python
WINNER_CONDITION = "B_simclr"  # or whichever won
```

```python
exec(open("/kaggle/working/CV/notebooks/kaggle_finetune_kfold.py").read())
```

Trains 5 folds (~45 min each) then runs TTA + ensemble inference on test set.

### 4.8 Output
- `/kaggle/working/submission_<WINNER>_5fold.csv` — submit to Kaggle
- 5 bundles `demo_bundle_<WINNER>_fold[0-4].pth`

### 4.9 Submit
Notebook → **Submit to Competition** → select CSV → wait for public LB score.

Target: public LB log loss < 0.5 (typical SSL/ImageNet ResNet-18 range).

---

## Day 5 — Local demo (~1h)

**Goal:** OOD generalization demo with Grad-CAM. Local machine, CPU OK.

### 5.1 Local setup
```bash
git pull
python -m venv .venv
.venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

### 5.2 Download bundles
From Day 4 PM notebook output, download into `outputs/finetune/`:
- `demo_bundle_A_scratch_fold0.pth`
- `demo_bundle_B_simclr_fold0.pth`
- `demo_bundle_C_imagenet_fold0.pth`

### 5.3 OOD images
Create `demo_images/` with ~10 Google-sourced driver photos NOT from State Farm dataset (different cars, lighting, angles).

### 5.4 Run demo
```bash
python notebooks/demo.py
```

Produces per-image 3-row figures: image + class prob bars + Grad-CAM overlay for all 3 models.

### 5.5 Qualitative assessment
- Does B_simclr or C_imagenet generalize better OOD?
- Where does Grad-CAM look? (hands? phone? face?)
- Document findings in `PLAN.md` § 10.

---

## Common gotchas

### Kaggle
- **Session timeout 9h interactive.** Save ckpt early, not at the wire.
- **"Save Version" runs a fresh kernel** — does NOT continue your interactive state. Use download-then-reupload pattern.
- **Internet defaults Off** for competition notebooks. Enable in Settings.
- **`/kaggle/working` wiped between sessions.** Always download ckpts you want to keep.
- **Competition data mounts at `/kaggle/input/competitions/<slug>/`** on newer Kaggle. Older docs say `/kaggle/input/<slug>/`. Verify with `!ls /kaggle/input/`.

### Training
- **DDP world_size=2 → per-GPU batch 128.** Effective batch still 256 because NT-Xent gathers across ranks.
- **Do NOT use gradient accumulation** to fake larger batch — breaks NT-Xent in-batch negatives.
- **No horizontal flip** in aug — would mix "phone right" with "phone left" classes.
- **GroupKFold by subject** is non-negotiable. Random split = driver leakage = inflated val acc.

### Reproducibility
- Seed = 24521897 everywhere (`src/seed_utils.py`)
- DDP uses same seed across ranks but `DistributedSampler` shards by rank
- If results differ between runs, check: aug RNG, DataLoader `worker_init_fn`, AMP nondeterminism (small)

---

## Reference: file pointers

| Task | File | Notes |
|---|---|---|
| Constants/paths | `src/config.py` | Change `KAGGLE_INPUT` if mount differs |
| Pretrain entry | `src/pretrain.py` | DDP-aware, `mp.spawn` from notebook |
| Finetune entry | `src/finetune.py` | 2-stage (freeze → discriminative LR) |
| Submission | `src/submit.py` | TTA 5-crop, no flip, clip [1e-15, 1-1e-15] |
| Augmentation | `src/augmentation.py` | NO horizontal flip |
| Group splits | `src/data.py` | `build_group_kfold(df)` |
| NT-Xent loss | `src/loss.py` | `gather_distributed=True` for DDP |

---

## Checklist (tick as you go)

- [ ] Day 1: sanity check passes, no fold leakage
- [ ] Day 2: pretrain to ep200 (or partial ckpt uploaded if timeout)
- [ ] Day 3: pretrain resume to ep200 (skip if Day 2 finished)
- [ ] `simclr-pretrain-final` dataset uploaded
- [ ] Day 4 PM: 3 conditions trained fold 0, headline recorded
- [ ] Day 4 Late: 5-fold ensemble of winner, submission CSV generated
- [ ] Submitted to Kaggle, public LB score recorded
- [ ] Day 5: local demo run, OOD figures saved
- [ ] `PLAN.md` § 10 reporting table filled
