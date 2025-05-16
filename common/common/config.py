from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data"

HQ50K_DIR = DATA_DIR / "HQ-50K"
IDPA_DIR = DATA_DIR / "image-deblurring-performance-analysis"

# HQ-50k/
HQ50K_TEST_DIR = HQ50K_DIR / "test"
HQ50K_TRAIN_DIR = HQ50K_DIR / "train"
HQ50K_TRAIN_DATASET = HQ50K_TRAIN_DIR / "all.txt"
HQ50K_TRAIN_SAMPLE_DATASET = HQ50K_TRAIN_DIR / "sample.parquet"

# image-deblurring-performance-analysis/
IDPA_DATASET = IDPA_DIR / "image_deblurring_dataset.parquet"

# train/
TRAIN_DIR = IDPA_DIR / "train"
TRAIN_ORIGINAL_DIR = TRAIN_DIR / "original"
TRAIN_BLURRED_DIR = TRAIN_DIR / "blurred"

# weights/
WEIGHTS_DIR = IDPA_DIR / "weights"
OFFICIAL_WEIGHTS_DIR = WEIGHTS_DIR / "official"
TRAIN_WEIGHTS_DIR = WEIGHTS_DIR / "train"

OFFICIAL_DEBLURGANV2_WEIGHTS = OFFICIAL_WEIGHTS_DIR / "deblurganv2_weights.h5"
OFFICIAL_MPRNET_WEIGHTS = OFFICIAL_WEIGHTS_DIR / "mprnet_weights.pth"

TRAIN_DEBLURGANV2_WEIGHTS = TRAIN_WEIGHTS_DIR / "deblurganv2_weights.h5"
TRAIN_MPRNET_WEIGHTS = TRAIN_WEIGHTS_DIR / "mprnet_weights.pth"

# test/
TEST_DIR = IDPA_DIR / "test"
TEST_ORIGINAL_DIR = TEST_DIR / "original"
TEST_BLURRED_DIR = TEST_DIR / "blurred"
TEST_DEBLURRED_DIR = TEST_DIR / "deblurred"

# BLUR PARAMETERS
BLUR_PARAM_RANGES = {
    'box': {'size': (5, 15, int)},
    'gaussian': {'size': (5, 21, int),'sigma': (0.5, 4.0, float)},
    'motion': {'length': (5, 30, int), 'angle': (0, 360, float)},
}