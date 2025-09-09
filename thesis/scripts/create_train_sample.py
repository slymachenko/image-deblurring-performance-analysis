from pathlib import Path
import pandas as pd
import random
import argparse

from common.config import HQ50K_TRAIN_DATASET, HQ50K_TRAIN_SAMPLE_DATASET

def load_sample_urls(file_path, sample_size):
    if not isinstance(file_path, Path):
        if isinstance(file_path, str):
            file_path = Path(file_path)
        else:
            raise TypeError("file_path must be a string or a pathlib.Path object")

    reservoir = []
    with file_path.open("r") as file:
        for i, line in enumerate(file):
            if i < sample_size:
                reservoir.append(line)
            else:
                j = random.randint(0, i)
                if j < sample_size:
                    reservoir[j] = line

    if len(reservoir) < sample_size:
        raise ValueError("Sample size cannot be greater than the number of URLs in the file")

    return reservoir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create Train Sample')
    parser.add_argument('--size', default=5000, type=int, help='Sample Size')
    parser.add_argument('--input', default=HQ50K_TRAIN_DATASET, type=int, help='Path to the Input Dataset')
    parser.add_argument('--output', default=HQ50K_TRAIN_SAMPLE_DATASET, type=int, help='Path to the Output Sample Dataset')

    args = parser.parse_args()
    sample_size = args.size
    input_path = args.input
    output_path = args.output

    sample_urls = load_sample_urls(input_path, sample_size)

    sample_urls_df = pd.DataFrame(sample_urls, columns=["url"])
    sample_urls_df.to_parquet(output_path, index=False)