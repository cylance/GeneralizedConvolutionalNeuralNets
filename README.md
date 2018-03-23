Copyright 2017 Cylance Inc.

# Generalized Convolutional Neural Networks for Point Cloud Data
This repo contains the code used for experiments described in our paper [Generalized Convolutional Neural Networks for Point Cloud Data](https://arxiv.org/abs/1707.06719), presented by Aleksandr Savchenkov during the poster session at the IEEE International Conference On Machine Learning and Applications (ICMLA 2017).

The goal of this work was to find a way to apply convolutional neural networks to point cloud data with arbitrary spatial features. By creating a mapping of nearest neighbors in a dataset, and applying a small shared neural network to the spatial relationships between points, we achieve an architecture that works directly with point clouds, but closely resembles a convolutional neural net in both design and behavior. Such a method bypasses the need for extensive feature engineering, while proving to be computationally efficient and requiring few parameters.

## Getting Started
### Requirements
* [numpy](http://www.numpy.org/)
* [TensorFlow](https://www.tensorflow.org)
* [trimesh](https://github.com/mikedh/trimesh) (for vectorization)
* [pyflann](https://github.com/primetang/pyflann) (for nearest neighbor computation)

## Usage
1. Vectorize:
    ```sh
    ./vectorize.sh
    ```
    [vectorize.sh](./vectorize.sh) uses the hyperparameters used in the paper.
2. Train:
    ```python
    python train.py
    ```
    Between epochs 10 and 20 you should see test performance hit between 92.2% and 92.8% (there is some randomness).
