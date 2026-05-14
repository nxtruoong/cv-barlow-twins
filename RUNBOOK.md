# Project Runbook — Barlow Twins + ResNet-18 on State Farm

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

## Day 2 — Barlow Twins pretrain (session 1: ~5h, epochs 1–40)

**Goal:** train BT for 40 epochs in session 1. Resume next session for 41–80.

**Timing math:**
- 399 steps/epoch × ~1.0s/step ≈ 6.7 min/epoch (SyncBN + 2048³ projector overhead)
- 40 epochs × 6.7 min ≈ 4.5h + ~10 min linear probe at ep 20/40
- Kaggle interactive cap ~9h → comfortable

**Mid-run sanity:** after epoch 1 stabilizes, check tqdm `it/s` and postfix:
- ≥ 1 it/s → on track
- < 0.5 it/s → data loader bottleneck. Stop, bump num_workers=4, restart
- `diag` field should rise toward 1.0; `off` field should fall below 0.2 by ep 10

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
- tqdm shows per-epoch loss + `diag`, `off` postfix every 20 steps
- Every epoch the log_entry records `c_diag_mean` and `c_offdiag_rms`
- Linear probe runs at epochs 20 and 40 in this session
- Checkpoint saved every 10 epochs to `/kaggle/working/bt/`

**Healthy trajectory:**
| Epoch | Loss | c_diag_mean | c_offdiag_rms | probe_acc |
|---|---|---|---|---|
| 0 | ~2000 | ~0.05 | ~0.4 | — |
| 10 | ~500 | ~0.4 | ~0.2 | — |
| 20 | ~200 | ~0.6 | ~0.12 | ~0.35 |
| 40 | ~80 | ~0.8 | ~0.08 | ~0.55 |

**Pre-registered abort criteria (epoch 20):** if `probe_acc < 0.25` AND `c_diag_mean < 0.3` AND `c_offdiag_rms > 0.4` → start Condition A scratch fine-tune in parallel; let BT run to completion and report outcome honestly (per ADR-0003).

### 2.5 End of session (epoch 40 reached)
1. Verify checkpoint:
   ```python
   !ls -la /kaggle/working/bt/
   ```
   Expect `bt_resnet18_ep039.pth` + `bt_resnet18_latest.pth` + `history.json`.
2. Download `bt_resnet18_latest.pth`
3. Upload as Kaggle Dataset `bt-ckpt-resume`
4. Go to Day 3 (resume for epochs 41–80)

---

## Day 3 — Barlow Twins pretrain (session 2: epochs 41–80, ~5h)

### 3.1 New notebook
- GPU T4 x2, Internet On
- Attach inputs:
  - State Farm Distracted Driver Detection (competition)
  - `bt-ckpt-resume` (Day 2 partial ckpt)

### 3.2 Clone + set resume path
```python
!git clone https://github.com/nxtruoong/CV.git /kaggle/working/CV
!pip install -q timm
```

Edit `notebooks/kaggle_pretrain.py` `RESUME` line before exec, OR inline:
```python
import sys, os, argparse
REPO = "/kaggle/working/CV"
sys.path.insert(0, REPO)
os.environ["PYTHONPATH"] = REPO + os.pathsep + os.environ.get("PYTHONPATH", "")

import torch, torch.multiprocessing as mp
from src.pretrain import run_pretrain, ddp_worker
from src.config import BT_BATCH_SIZE, BT_EPOCHS, BT_LR, BT_LAMBDA

RESUME = "/kaggle/input/bt-ckpt-resume/bt_resnet18_latest.pth"
args = argparse.Namespace(
    epochs=BT_EPOCHS, batch_size=BT_BATCH_SIZE, lr=BT_LR,
    lambda_off=BT_LAMBDA, num_workers=2, amp=True,
    save_every=10, resume=RESUME,
    output_dir="/kaggle/working/bt",
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
Resumed from /kaggle/input/bt-ckpt-resume/bt_resnet18_latest.pth at epoch <N>
```

### 3.4 End of session
Download `bt_resnet18_ep079.pth` + `history.json`. Upload as `bt-pretrain-final`.

---

## Day 4 PM — Fine-tune 3 conditions, fold 0 (~3h)

**Goal:** A_scratch vs B_bt vs C_imagenet on fold 0. Pick winner by val log loss.

### 4.3 New notebook
- GPU T4 x2 (only 1 used, but parallel sessions allowed)
- Attach: competition data + `bt-pretrain-final`

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
  B_bt:   val_log_loss = 0.XXXX
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
WINNER_CONDITION = "B_bt"  # or whichever won
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
- `demo_bundle_B_bt_fold0.pth`
- `demo_bundle_C_imagenet_fold0.pth`

### 5.3 OOD images
Create `demo_images/` with ~10 Google-sourced driver photos NOT from State Farm dataset (different cars, lighting, angles).

### 5.4 Run demo
```bash
python notebooks/demo.py
```

Produces per-image 3-row figures: image + class prob bars + Grad-CAM overlay for all 3 models.

### 5.5 Qualitative assessment
- Does B_bt or C_imagenet generalize better OOD?
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
- **DDP world_size=2 → per-GPU batch 128.** Effective batch still 256 because BT all-reduces the local outer product across ranks before computing C.
- **SyncBatchNorm must be applied BEFORE wrapping in DDP.** `src/pretrain.py` does this via `BarlowTwinsModel.convert_sync_bn(model)`. Without it, the final projector BN normalizes per-GPU → per-GPU C matrix → silent undertraining.
- **Do NOT use gradient accumulation** to fake larger batch — desynchronizes BN/C-matrix statistics from the optimizer step.
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
| Pretrain entry | `src/pretrain.py` | Barlow Twins, DDP-aware, `mp.spawn` from notebook |
| Finetune entry | `src/finetune.py` | 2-stage (freeze → discriminative LR) |
| Submission | `src/submit.py` | TTA 5-crop, no flip, clip [1e-15, 1-1e-15] |
| Augmentation | `src/augmentation.py` | NO horizontal flip |
| Group splits | `src/data.py` | `build_group_kfold(df)` |
| Barlow Twins loss | `src/loss.py` | `BarlowTwinsLoss` + `dist.all_reduce` of local outer product |
| Phase 1 ADR | `docs/adr/0003-barlow-twins-over-simclr.md` | Method choice, projector dim, fallback policy |

---

## Checklist (tick as you go)

- [ ] Day 1: sanity check passes, no fold leakage
- [ ] Day 2: BT pretrain to ep40 (session 1); partial ckpt uploaded
- [ ] Day 3: BT pretrain resume to ep80 (session 2)
- [ ] Epoch 20 abort criteria checked (probe/diag/off-rms thresholds)
- [ ] `bt-pretrain-final` dataset uploaded
- [ ] Day 4 PM: 3 conditions trained fold 0, headline recorded
- [ ] Day 4 Late: 5-fold ensemble of winner, submission CSV generated
- [ ] Submitted to Kaggle, public LB score recorded
- [ ] Day 5: local demo run, OOD figures saved
- [ ] `PLAN.md` § 10 reporting table filled
