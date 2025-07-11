# Brain Tumor Classification using HOG and SVM

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0%2B-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Project Description
This repository contains a complete machine learning pipeline for classifying brain tumor MRI images into four categories:
- Glioma
- Meningioma
- No tumor
- Pituitary tumor

The system uses Histogram of Oriented Gradients (HOG) for feature extraction combined with color histograms, and evaluates Support Vector Machines (SVM) with different kernels for classification.

## Features
- **Image Preprocessing**: Automatic resizing and color conversion
- **Feature Extraction**: HOG + Color Histograms
- **Model Training**: SVM with 3 kernel types (Linear, Polynomial, RBF)
- **Hyperparameter Tuning**: Automated grid search for optimal parameters
- **Evaluation Metrics**:
  - Classification reports (precision, recall, F1-score)
  - Confusion matrices
  - ROC curves (for probabilistic models)
  - Learning curves
