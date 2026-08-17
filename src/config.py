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

# BERT text encoder
bert_model_name = os.getenv("BERT_MODEL_NAME", "bert-base-multilingual-cased")
bert_feature_dim = int(os.getenv("BERT_FEATURE_DIM", "768"))

# Image preprocessing
image_size = int(os.getenv("IMAGE_SIZE", "125"))
crop_scale = tuple(float(v) for v in os.getenv("CROP_SCALE", "0.1,1").split(","))
crop_ratio = tuple(float(v) for v in os.getenv("CROP_RATIO", "0.5,2").split(","))
norm_mean = tuple(float(v) for v in os.getenv("NORM_MEAN", "0.485,0.456,0.406").split(","))
norm_std = tuple(float(v) for v in os.getenv("NORM_STD", "0.229,0.224,0.225").split(","))

# Train/test split
split_random_state = int(os.getenv("SPLIT_RANDOM_STATE", "0"))

# DataLoader worker processes (Windows spawns workers, so the default there is 0)
num_workers = int(os.getenv("NUM_WORKERS", "0" if os.name == "nt" else "4"))

# Model head widths
fusion_hidden_dim = int(os.getenv("FUSION_HIDDEN_DIM", "64"))
image_feature_dim = int(os.getenv("IMAGE_FEATURE_DIM", "512"))
meta_feature_dim = int(os.getenv("META_FEATURE_DIM", "3"))
dropout_rate = float(os.getenv("DROPOUT_RATE", "0.5"))

# Prediction head sizes
num_best_sex_class = int(os.getenv("NUM_BEST_SEX_CLASS", "3"))
num_best_age_class = int(os.getenv("NUM_BEST_AGE_CLASS", "7"))
num_sales_class = int(os.getenv("NUM_SALES_CLASS", "7"))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
data_path = Path(os.getenv("DATA_PATH", str(PROJECT_ROOT / "dataset" / "category_all_ver2_20221002_words_125_aug")))
csv_path = Path(os.getenv("CSV_PATH", str(PROJECT_ROOT / "dataset" / "goodsNum_clothing_name_20221002.csv")))

CHECKPOINT_DIR = Path(os.getenv("CHECKPOINT_DIR", str(PROJECT_ROOT / "checkpoints")))
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", str(PROJECT_ROOT / "results")))

def ensure_directories_exist() -> None:
    """Create required directories if they don't exist."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)