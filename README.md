# Tensor Train Image Classification

This project explores image classification using tensor-based numerical methods, with a focus on **Tensor Train (TT) decomposition**, **kernelized MANDy**, and **Alternating Ridge Regression (ARR)**.

The goal is to show how high-dimensional feature spaces can be used for supervised image classification without explicitly constructing and storing the full feature representation.

The project was developed as part of a university seminar on tensor decompositions and their applications in image classification.

## Project Overview

Classical image classification tasks often involve high-dimensional data. For example, MNIST and Fashion-MNIST images have size `28 × 28`, which means each image can be represented as a vector in `R^784`.

This project studies how tensor methods can be used to map image data into very high-dimensional feature spaces while keeping computations feasible through low-rank tensor representations.

The main focus is on:

- Tensor Train decomposition
- Kernelized MANDy method
- Alternating Ridge Regression
- MNIST classification
- Fashion-MNIST classification
- Analysis of misclassified examples and confusion matrices

## Motivation

Tensor methods are useful when working with high-dimensional data because they can reduce memory and computational complexity.

Instead of explicitly storing exponentially large feature vectors, this project uses tensor product structures and Tensor Train representations to make the classification problem computationally manageable.

This approach is especially interesting from the perspective of:

- numerical linear algebra
- scientific computing
- machine learning
- image classification
- high-dimensional approximation

## Datasets

The experiments are based on:

- **MNIST** — handwritten digit classification
- **Fashion-MNIST** — clothing item classification

Each image has size `28 × 28` pixels and is represented as a vector of length `784`.

The classification task has 10 possible output classes, represented using one-hot encoding.

## Methods

### Tensor Train Decomposition

Tensor Train decomposition is used to represent high-dimensional tensors in a compressed format.

Instead of storing all tensor entries explicitly, the tensor is represented as a product of smaller core tensors. This reduces the number of parameters and makes high-dimensional computations feasible.

### Kernelized MANDy

The kernelized MANDy method maps input data into a high-dimensional feature space using tensor product basis functions.

Instead of explicitly constructing the full feature matrix, the method works with a Gram matrix and a kernel representation.

This allows classification to be performed in an implicitly defined high-dimensional space.

### Alternating Ridge Regression

Alternating Ridge Regression directly approximates the coefficient tensor in Tensor Train format.

The optimization is performed by updating one TT core at a time while keeping the others fixed. This approach is inspired by ALS/DMRG-type sweeping methods.

Compared to the kernelized approach, ARR can reduce memory usage, but it depends on the choice of TT ranks, regularization, and the number of optimization sweeps.

## Experiments

The methods were tested on subsets of MNIST and Fashion-MNIST.

The experiments include:

- learning curves for different training set sizes
- comparison between kernelized MANDy and ARR
- visualization of misclassified samples
- confusion matrices for classification errors
- discussion of differences between MNIST and Fashion-MNIST performance

## Results

The results show that the kernelized MANDy method performs very well on MNIST and generally achieves higher accuracy than ARR.

Fashion-MNIST is more challenging because many classes have visually similar shapes, such as shirts, coats, pullovers, and other clothing items.

The confusion matrices show that most errors occur between visually similar classes, which suggests that the tensor-based models capture global image structure but may struggle with fine local details.

## Key Takeaways

- Tensor methods can be applied to supervised image classification.
- Tensor Train decomposition helps manage high-dimensional feature spaces.
- Kernelized MANDy provides a direct dual solution using a Gram matrix.
- ARR offers a low-rank Tensor Train approach with lower memory requirements.
- MNIST is easier to classify than Fashion-MNIST due to clearer class separation.
- Most classification errors occur between visually similar classes.

## Technologies

Possible implementation tools:

- Python
- NumPy
- SciPy
- scikit-learn
- Matplotlib
- PyTorch
- TensorLy

## Repository Structure

```text
.
├── README.md
├── src/
│   ├── data_preprocessing.py
│   ├── kernel_mandy.py
│   ├── alternating_ridge_regression.py
│   ├── evaluation.py
│   └── visualization.py
├── notebooks/
│   └── experiments.ipynb
├── results/
│   ├── learning_curves/
│   ├── confusion_matrices/
│   └── misclassified_examples/
├── report/
│   └── seminar.pdf
└── requirements.txt
