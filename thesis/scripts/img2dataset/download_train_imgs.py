from img2dataset import download
from pathlib import Path

from common.config import HQ50K_TRAIN_SAMPLE_DATASET, TRAIN_ORIGINAL_DIR

def download_images(url_list, output_dir, cpu_cores=1):
    if not isinstance(url_list, Path):
        if isinstance(url_list, str):
            url_list = Path(url_list)
        else:
            raise TypeError("url_list must be a string or a pathlib.Path object")

    if not isinstance(output_dir, Path):
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)
        else:
            raise TypeError("output_dir must be a string or a pathlib.Path object")

    output_dir.mkdir(parents=True, exist_ok=True)

    download(
        url_list=str(url_list),
        input_format="parquet",
        output_folder=str(output_dir),
        resize_mode="no",
        encode_format="png",
        encode_quality=9,
        processes_count=cpu_cores,
        thread_count=cpu_cores*2,
        incremental_mode="incremental"
    )

if __name__ == "__main__":
    download_images(
        url_list=HQ50K_TRAIN_SAMPLE_DATASET,
        output_dir=TRAIN_ORIGINAL_DIR,
        cpu_cores=8
    )