"""Configuration with environment variable support"""
import os
from pathlib import Path
from typing import Optional, Union

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

CUDA_VISIBLE_DEVICES = int(os.getenv("CUDA_VISIBLE_DEVICES", "0"))
batch_size = int(os.getenv("BATCH_SIZE", "64"))
num_epochs = int(os.getenv("NUM_EPOCHS", "3000"))
lr = float(os.getenv("LR", "0.0001"))
weight_decay = float(os.getenv("WEIGHT_DECAY", "1e-5"))
clip_norm = float(os.getenv("CLIP_NORM", "5"))

# focal loss weight
alpha = float(os.getenv("FOCAL_ALPHA", "1"))
gamma = float(os.getenv("FOCAL_GAMMA", "2"))

# loss weight
loss_alpha = float(os.getenv("LOSS_ALPHA", "0.01"))
loss_beta = float(os.getenv("LOSS_BETA", "0.01"))

PROJECT_ROOT = Path(__file__).parent.parent.parent
data_path = Path(os.getenv("DATA_PATH", str(PROJECT_ROOT / "dataset" / "category_all_ver2_20221002_words_125_aug")))
csv_path = Path(os.getenv("CSV_PATH", str(PROJECT_ROOT / "dataset" / "goodsNum_clothing_name_20221002.csv")))

CHECKPOINT_DIR = Path(os.getenv("CHECKPOINT_DIR", str(PROJECT_ROOT / "checkpoints")))
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", str(PROJECT_ROOT / "results")))

def ensure_directories_exist() -> None:
    """Create required directories if they don't exist."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)