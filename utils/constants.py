from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parent.parent

INIT_IMAGES_DATASET_PATH = ROOT_PATH / "data/HQ-50k/test"
DIR_DATASET_PATH = ROOT_PATH / "data/image-deblurring-performance-analysis"

MAIN_DATASET_PATH = DIR_DATASET_PATH / "image_deblurring_dataset.parquet"
ORIGINAL_DATASET_PATH = DIR_DATASET_PATH / "original"
BLURRED_DATASET_PATH = DIR_DATASET_PATH / "blurred"
DEBLURRED_DATASET_PATH = DIR_DATASET_PATH / "deblurred"