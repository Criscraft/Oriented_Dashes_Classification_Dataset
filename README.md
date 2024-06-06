# Oriented Dashes Classification Dataset

## Description

This dataset contains 1024 gray-scale images featuring various horizontal and vertical dashes. The primary task associated with this dataset is to determine whether an image contains more horizontal dashes than vertical dashes. The resulting two classes 'horizontal' and 'vertical' form a binary classification problem. To obtain the dataset, download the images.zip file.

## Features

- Image Count: 1024 artificially generated images.
- Resolution: 48x48 pixels.
- Dash Characteristics: Each dash is 1 pixel thick and 5 pixels long.
- Orientation Classification: Images contain varying counts of horizontal and vertical dashes, with no images having an equal number of both.
- Gradient Information Utilization: Effective utilization of gradient information is essential for accurately solving the orientation classification problem.
- The code for generating the images can be adjusted to get additional images. The resolution and dash size can be varied to achieve different characteristics.

## Installation
To install the necessary requirements for this project, follow these steps:

1. Create a virtual environment using conda (optional but recommended):
    ```bash
    conda create --name myenv
    ```

2. Activate the virtual environment:
    ```bash
    conda activate myenv
    ```

3. Install PyTorch and other dependencies. For more details, see the [Pytorch](https://pytorch.org/) website.
    ```bash
    conda install torch matplotlib
    ```

4. Clone the repository:
    ```bash
    git clone https://github.com/your-username/your-repository.git
    ```

5. Change to the project directory:
    ```bash
    cd your-repository

## Usage
To generate the images using the `toy_dataset.py` script, follow these steps:

1. Make sure you have installed the necessary requirements as mentioned above.

2. Open a terminal and navigate to the project directory.

3. Run the following command to generate the images:
    ```bash
    python toy_dataset.py
    ```

4. The script will generate images with the specified characteristics and save them in a directory 'images'.

5. You can adjust the parameters in the `toy_dataset.py` script to modify the image count, resolution, or dash size.

The script `test_model.py` demonstrates how a simple model with pre-defined convolution filters can solve the dataset.


## License
This project is licensed under the MIT License.
