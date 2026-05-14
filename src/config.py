"""Project-wide constants and paths."""
from pathlib import Path

SEED = 24521897

IMG_SIZE = 224
PRETRAIN_IMG_SIZE = 224  # square — TPU v5e-8 plan, kills FixRes
PRETRAIN_IMG_H = 224
PRETRAIN_IMG_W = 224
PRETRAIN_CACHE_RES = 256  # uint8 memmap shape: (N, 256, 256, 3)
PRETRAIN_CACHE_PATH = "/tmp/pretrain_cache.bin"
PRETRAIN_CACHE_INDEX = "/tmp/pretrain_cache_index.json"
NUM_CLASSES = 10

CLASS_NAMES = [
    "safe driving",
    "texting - right",
    "phone - right",
    "texting - left",
    "phone - left",
    "operating radio",
    "drinking",
    "reaching behind",
    "hair/makeup",
    "talking to passenger",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

PRETRAIN_BATCH_SIZE = 2048  # global, 256/core × 8 chips on TPU v5e-8
PRETRAIN_PER_CORE_BATCH = 256
PRETRAIN_EPOCHS = 100
PRETRAIN_WARMUP_EPOCHS = 10
PRETRAIN_LR = 2.828e-3  # 1e-3 * sqrt(2048/256), sqrt-rule LR scale for AdamW
PRETRAIN_WEIGHT_DECAY = 1e-4
NT_XENT_TEMPERATURE = 0.5
PROJECTION_DIM = 128
PROJECTION_HIDDEN = 512

FINETUNE_STAGE1_EPOCHS = 5
FINETUNE_STAGE2_EPOCHS = 25
FINETUNE_STAGE2_WARMUP = 2
FINETUNE_LABEL_SMOOTHING = 0.1
FINETUNE_EARLY_STOP_PATIENCE = 5
FINETUNE_BATCH_SIZE = 128

N_FOLDS = 5

KAGGLE_INPUT = Path("/kaggle/input/competitions/state-farm-distracted-driver-detection")
KAGGLE_WORKING = Path("/kaggle/working")

LOCAL_DATA = Path("data/state-farm-distracted-driver-detection")
LOCAL_WORKING = Path("outputs")


def get_data_root() -> Path:
    if KAGGLE_INPUT.exists():
        return KAGGLE_INPUT
    return LOCAL_DATA


def get_working_dir() -> Path:
    if KAGGLE_WORKING.exists():
        return KAGGLE_WORKING
    LOCAL_WORKING.mkdir(parents=True, exist_ok=True)
    return LOCAL_WORKING
