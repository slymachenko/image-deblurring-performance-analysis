# image-deblurring-performance-analysis

> [!WARNING]
> You might encounter issues, as this project is in active development.

**Name:** Statistical Analysis of Image Deblurring Methods Performance Across Classical and AI-Based Approaches  

In the field of image processing, the degradation of image quality due to various types of distorions poses significant challenges for both human interpretation and machine learning applications. This project aims to systematically investigate blur image distortions by statistically analyzing relationships between parameters of the blurred images and metrics of the deblurring methods, both classical and AI-based.

## Getting Started

> [!NOTE]
> Please make sure you have [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) installed on your system.

1. Clone the Repository:

    ```bash
    git clone https://github.com/slymachenko/image-deblurring-performance-analysis.git
    cd image-deblurring-performance-analysis
    ```

2. Create Conda environment using file `environment.yml`:

    ```bash
    conda create --name image_deblurring_performance_analysis --file environment.yml
    conda activate image_deblurring_performance_analysis
    ```

3. Run the desired script/notebook in the conda environment.

## Dataset creation

To create our dataset we took a test subset (1250 images) of [HQ-50K: A Large-scale, High-quality Dataset for Image Restoration](https://github.com/littleYaang/HQ-50K).

The project is done assuming the following structure of the `data/` directory:

```text
.
├── data/
│   ├── HQ-50k/
│   │   ├── test/
│   │   └── train
│   ├── image_deblurring_dataset.csv
│   └── images/
│       ├── blurred/
│       ├── deblurred/
│       └── orignal/
```

where:

- `HQ-50k/`: [HQ-50K: A Large-scale, High-quality Dataset for Image Restoration](https://github.com/littleYaang/HQ-50K)
- `image_deblurring_dataset.csv`: Dataset that contains the following: Image Metadata, Blur Parameters, Performance Metrics
- `images/`: Downloaded images, separated into its category (original, blurred, deblurred)

## License

The project is released under the [MIT license](LICENSE).
