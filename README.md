# image-deblurring-performance-analysis

**Name:** Statistical Analysis of Image Deblurring Methods Performance Across Classical and AI-Based Approaches  

In the field of image processing, the degradation of image quality due to various types of distorions poses significant challenges for both human interpretation and machine learning applications. This project aims to systematically investigate blur image distortions by statistically analyzing relationships between parameters of the blurred images and metrics of the deblurring methods, both classical and AI-based.

Dataset is availiable on [[HuggingFace]](https://huggingface.co/datasets/slymachenko/image-deblurring-performance-analysis)

## Getting Started

> [!NOTE]
> Please make sure you have [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) installed on your system.

1. Clone the Repository:

    ```bash
    git clone https://github.com/slymachenko/image-deblurring-performance-analysis.git
    cd image-deblurring-performance-analysis
    ```

2. Create Conda environment using file `environment.yml`:

   - For *Dataset Creation* part:

    ```bash
    conda create --file notebooks/dataset_creation/environment.yml
    conda activate idpa-creation
    ```

   - For *Dataset Analysis* part:

    ```bash
    conda create --file notebooks/dataset_analysis/environment.yml
    conda activate idpa-analysis
    ```

3. Download dataset from HuggingFace: [[Link]](https://huggingface.co/datasets/slymachenko/image-deblurring-performance-analysis)

    Follow the file structure of the HuggingFace repository with the root of it being `data/image-deblurring-performance-analysis/`

4. Run the desired script/notebook in the conda environment.

## Credits

Parts of this project were inspired by or copied from the following sources:

- [MPRNet repository](https://github.com/swz30/MPRNet)
- [DeblurGAN-v2 repository](https://github.com/VITA-Group/DeblurGANv2)

## License

The project is released under the [MIT license](LICENSE).
