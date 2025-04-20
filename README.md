# Brain Tumor Classification using HOG and SVM

This repository contains Python code for classifying brain tumors from MRI images. It utilizes the Histogram of Oriented Gradients (HOG) for feature extraction and Support Vector Machines (SVM) for classification into four categories: glioma, meningioma, notumor, and pituitary.

## Overview

The project is structured into the following Python files:

-   `data_loader.py`: Handles loading and preprocessing of image data from a specified directory structure.
-   `feature_extractor.py`: Implements the HOG feature extraction method.
-   `model.py`: Defines and trains the SVM classification model with different kernel options.
-   `evaluation.py`: Contains functions for evaluating the trained model, including generating classification reports, confusion matrices, and accuracy plots.
-   `main.py`: The main script that orchestrates the data loading, feature extraction, model training, prediction, and evaluation process.

## Dataset

The code assumes a dataset directory named `dataset` in the same root directory as the scripts. This directory should be organized as follows:
