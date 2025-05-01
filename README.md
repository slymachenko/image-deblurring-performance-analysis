# image-deblurring-performance-analysis

> [!WARNING]
> You might encounter issues, as this project is in active development.

**Name:** Statistical Analysis of Image Deblurring Methods Performance Across Classical and AI-Based Approaches  

In the field of image processing, the degradation of image quality due to various types of distorions poses significant challenges for both human interpretation and machine learning applications. This project aims to systematically investigate blur image distortions by statistically analyzing relationships between parameters of the blurred images and metrics of the deblurring methods, both classical and AI-based.

## Getting Started

### Dataset

Dataset is availiable on [[HuggingFace]](https://huggingface.co/datasets/slymachenko/image-deblurring-performance-analysis)

### Dataset creation and analysis

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

3. Download dataset from HuggingFace: [[Link]](https://huggingface.co/datasets/slymachenko/image-deblurring-performance-analysis)

4. Run the desired script/notebook in the conda environment.

## Dataset creation

To create our dataset we used a test subset (1250 images) of [HQ-50K: A Large-scale, High-quality Dataset for Image Restoration](https://github.com/littleYaang/HQ-50K).

## License

The project is released under the [MIT license](LICENSE).
