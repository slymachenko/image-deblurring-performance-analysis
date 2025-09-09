from img2dataset import download
from pathlib import Path
import argparse

from common.config import IDPA_DATASET, TEST_ORIGINAL_DIR

def download_images(url_list, output_dir):
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
        output_format="parquet",
        resize_mode="no",
        encode_format="png",
        encode_quality=9,
        thread_count=64,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download Test Images')
    parser.add_argument('--input', default=IDPA_DATASET, type=int, help='Path to the dataset.parquet with the list of urls')
    parser.add_argument('--output', default=TEST_ORIGINAL_DIR, type=int, help='Path to the output directory where images will be saved')

    args = parser.parse_args()
    url_list = args.input
    output_dir= args.output

    download_images(
        url_list=url_list,
        output_dir=output_dir,
    )