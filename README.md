# SimCLR + ResNet-18 on State Farm Distracted Driver Detection

Experimental study comparing self-supervised SimCLR pretraining vs random init vs ImageNet pretrain for the [State Farm Distracted Driver Detection](https://www.kaggle.com/c/state-farm-distracted-driver-detection) Kaggle competition. Designed to run end-to-end on free Kaggle T4 x2 sessions.

See [PLAN.md](PLAN.md) for full experimental design, hypotheses, and rationale.

## Repo Layout

```
.
├── PLAN.md                          # full experimental design
├── requirements.txt
├── src/
│   ├── config.py                    # constants, seed, paths
│   ├── seed_utils.py                # reproducibility
│   ├── augmentation.py              # pretrain + finetune aug (NO horizontal flip)
│   ├── data.py                      # GroupKFold by subject; datasets
│   ├── model.py                     # ResNet-18 + projection head / classifier
│   ├── loss.py                      # NT-Xent + DDP all_gather with gradient
│   ├── diagnostics.py               # linear probe + alignment/uniformity
│   ├── pretrain.py                  # SimCLR training (DDP-aware)
│   ├── finetune.py                  # Stage 1 (freeze) + Stage 2 (discriminative LR)
│   └── submit.py                    # TTA + K-fold ensemble + Kaggle CSV
└── notebooks/
    ├── kaggle_sanity_check.py       # Day 1
    ├── kaggle_pretrain.py           # Day 2-4 AM (DDP via mp.spawn)
    ├── kaggle_finetune.py           # Day 4: A vs B vs C (1 fold each)
    ├── kaggle_finetune_kfold.py     # Day 4 PM: 5-fold of winner
    └── demo.py                      # Day 5: OOD demo with Grad-CAM
```

## Quick Reference

| Setting | Value |
|---|---|
| Seed | 24521897 (everywhere) |
| Image size | 224×224 |
| Pretrain batch | 256 (128 per GPU on T4 x2) |
| Pretrain epochs | 200 (10 warmup + 190 cosine) |
| Optimizer | AdamW lr=1e-3 wd=1e-4 |
| NT-Xent τ | 0.5 |
| Fine-tune | Stage 1: 5 ep frozen / Stage 2: 25 ep discriminative LR + label smoothing 0.1 |
| Val | GroupKFold(5) by subject (NEVER random split) |
| TTA | 5 crops, NO horizontal flip |
| Submission | 5-fold ensemble of winner condition, clipped to [1e-15, 1-1e-15] |

## Running on Kaggle

### Day 1 — Sanity check (~30 min)
1. Create new Kaggle notebook
2. Add input: `state-farm-distracted-driver-detection` (competition data)
3. Add input: this repo (via Kaggle Dataset OR `!git clone` in cell)
4. Run `notebooks/kaggle_sanity_check.py`

### Day 2-4 — SimCLR pretraining (3 sessions × ~5h)
1. Set Accelerator: **GPU T4 x2**
2. First session: set `RESUME = None` in `kaggle_pretrain.py`, run
3. End of session: download `simclr_resnet18_latest.pth`, upload as new Kaggle Dataset
4. Next session: attach that dataset, set `RESUME = "/kaggle/input/<dataset>/simclr_resnet18_latest.pth"`, run
5. After epoch 200: keep final checkpoint as "simclr-pretrain-final" dataset

**Diagnostic monitoring:** Every 10 epochs the script logs linear probe accuracy + alignment/uniformity. If linear probe is stuck near 10% (random) by epoch 30, SSL is broken — stop and debug augmentation/temperature.

### Day 4 — Fine-tune all 3 conditions (~4-6h)
1. Attach `simclr-pretrain-final` dataset
2. Run `notebooks/kaggle_finetune.py` → trains A_scratch, B_simclr, C_imagenet on fold 0
3. Read headline log losses; pick winner
4. Set `WINNER_CONDITION` in `kaggle_finetune_kfold.py`, run → 5-fold ensemble + submission CSV

### Day 5 — Demo
1. Locally: clone repo, download all 3 fine-tune bundles
2. Put OOD Google images in `demo_images/`
3. Run `notebooks/demo.py` → 3-row figures per image (image + prob bars + Grad-CAM)

## Multi-GPU Notes

`pretrain.py` supports both single-GPU and DDP:
- Kaggle notebook with T4 x2: use `mp.spawn` (see `kaggle_pretrain.py`) — torchrun is awkward in notebooks
- Local machine with `torchrun`: `torchrun --nproc_per_node=N -m src.pretrain --epochs 200`
- Single GPU: `python -m src.pretrain --epochs 200`

The `NTXentLoss` class auto-detects distributed mode and gathers features with `all_gather` while preserving local-rank gradient. **Do not use gradient accumulation to fake larger batch** — it breaks NT-Xent.

## Success Criteria (pre-registered in PLAN.md)

- **H1**: SimCLR pretrain improves over random init (B better than A by ≥0.05 val log loss)
- **H2**: SimCLR competitive with ImageNet pretrain (B within +0.10 of C)
- **H3**: Combined approach value (B + C ensemble > C alone)

Fill in the reporting table in `PLAN.md` § 10 after experiments complete.

## License / Attribution

State Farm dataset © State Farm Mutual Automobile Insurance Company, used under Kaggle competition terms (test images used unlabeled for SSL — transductive learning, no label leakage).
