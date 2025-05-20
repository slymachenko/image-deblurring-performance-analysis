from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parent.parent
DIR_DATA_PATH = ROOT_PATH / "data"

INIT_IMAGES_DATASET_PATH = DIR_DATA_PATH / "HQ-50K" / "test"
DIR_DATASET_PATH = DIR_DATA_PATH / "image-deblurring-performance-analysis"

MAIN_DATASET_PATH = DIR_DATASET_PATH / "image_deblurring_dataset.parquet"
ORIGINAL_DATASET_PATH = DIR_DATASET_PATH / "original"
BLURRED_DATASET_PATH = DIR_DATASET_PATH / "blurred"
DEBLURRED_DATASET_PATH = DIR_DATASET_PATH / "deblurred"
WEIGHTS_PATH = DIR_DATASET_PATH / "weights"