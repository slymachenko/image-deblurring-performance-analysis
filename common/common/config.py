from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data"

HQ50K_DIR = DATA_DIR / "HQ-50K"
IDPA_DIR = DATA_DIR / "image-deblurring-performance-analysis"

DEMO_IMAGE_KEY = "000000845"

# HQ-50k/
HQ50K_TEST_DIR = HQ50K_DIR / "test"
HQ50K_TRAIN_DIR = HQ50K_DIR / "train"
HQ50K_TRAIN_DATASET = HQ50K_TRAIN_DIR / "all.txt"
HQ50K_TRAIN_SAMPLE_DATASET = HQ50K_TRAIN_DIR / "sample.parquet"

# image-deblurring-performance-analysis/
IDPA_DATASET = IDPA_DIR / "image_deblurring_dataset.parquet"
PSEUDOINVERSE_NOKERNEL_DATASET = IDPA_DIR / "pseudoinverse_nokernel.parquet"
PSEUDOINVERSE_WKERNEL_DATASET = IDPA_DIR / "pseudoinverse_wkernel.parquet"
WIENER_NOKERNEL_DATASET = IDPA_DIR / "wiener_nokernel.parquet"
WIENER_WKERNEL_DATASET = IDPA_DIR / "wiener_wkernel.parquet"
DEBLURGANV2_DATASET = IDPA_DIR / "deblurganv2.parquet"
MPRNET_DATASET = IDPA_DIR / "mprnet.parquet"

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

TEST_BLURRED_BOX_DIR = TEST_BLURRED_DIR / "box"
TEST_BLURRED_GAUSSIAN_DIR = TEST_BLURRED_DIR / "gaussian"
TEST_BLURRED_MOTION_DIR = TEST_BLURRED_DIR / "motion"

# BLUR PARAMETERS
BLUR_TYPES = ['box', 'gaussian', 'motion']
BLUR_PARAM_RANGES = {
    'box': {'size': (5, 15)},
    'gaussian': {'size': (5, 21),'sigma': (0.5, 4.0)},
    'motion': {'length': (5, 30), 'angle': (0, 360)},
}

# LISTS
METRIC_TYPES = ['proctime', 'ssim', 'psnr', 'lpips', 'gmsd']

# DEBLUR PARAMETERS
DEBLUR_MODELS = ['pseudo_inverse', 'wiener', 'deblurganv2', 'mprnet']

# IMAGRE FEATURES EXTRACTION PARAMETERS
GAUSSIAN_PARAMS = {
    'ksize': (3, 3),
    'sigma': 1.0
}

SOBEL_PARAMS = {
    'ksize': 3
}
