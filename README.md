# Project-UCL-ZiyanChen
This project was undertaken during the academic year 2022-2023 at University College London. The code and methodology in this repository are built upon and adapted from the original work of Zhenya Yan. Significant modifications and extensions have been made to suit the specific requirements and objectives of this project.

## Acknowledgment
A significant portion of the CNN-3D library utilized in this project is derived from the work by Alexander Whitehead, Institute of Nuclear Medicine, UCL.

**Copyright University College London 2022**

**Author: Alexander Whitehead, Institute of Nuclear Medicine, UCL**

**For internal research only.**

## Introduction
This project focuses on the study of positron annihilation and its interactions with Convolutional Neural Networks (CNN) training.

## Directory Structure and Contents

### `Code`
Contains all the code files used in the project.

- **`CNN`**: Contains both the original version by Zhenya Yan and the modified version for the implementation of the 3D CNN based on 3D-Unet. This directory includes `.ipynb` files to run on Google Colab and required library files.

- **`Density_Image`**: Generates random density maps (Converted to MuMap).

- **`LoadImages`**: Converts .txt files to .bin files.

- **`Voxelised_ZhenyaYan`**: Contains configuration files used by Zhenya Yan for GATE simulations when submitting to the cluster.

### `Output_ZhenyaYan`
Contains all the output files originally from Zhenya Yan's repository.

### `Output_ZiyanChen`
This directory consists of two main subdirectories:

- **`CNN_output`**: Contains results from various runs of the CNN under different conditions and settings.
  - Files:
    - `Anni_test.bin`: Annihilation (MC)
    - `MuMap_test.bin`: MuMap
    - `estim_test.bin`: Predicted annihilation image
    - PNG images representing the loss from the CNN runs.

- **`GATE_output`**: Contains four subdirectories, each with input, configuration, and output files related to GATE simulations.
  - **`Gate_adjust_test`**: Outputs from test runs during GATE configuration adjustments.
  - **`voxelization_ziyan_500`**: Results from simulations with 500 density maps under a 10 kBq radioactive activity setting.
  - **`voxelization_ziyan_2000`**: Results from simulations with 2000 density maps under a 100 kBq radioactive activity setting.
  - **`voxelization_ziyan_2000_1`**: Results from simulations with another set of 2000 density maps under a 100 kBq radioactive activity setting.
