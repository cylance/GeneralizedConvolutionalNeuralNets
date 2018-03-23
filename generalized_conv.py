# Utility functions and building blocks (generalized convolutional layers).

import tensorflow as tf
import numpy as np
from pyflann import *


def weight_variable(shape):
    initial = tf.random_normal(shape, stddev=1 / np.sqrt(shape[0]))
    return tf.Variable(initial, name='weight')


def bias_variable(shape):
    initial = tf.random_normal(shape, stddev=1 / np.sqrt(shape[0]))
    return tf.Variable(initial, name='bias')


def lrelu(x):
    return tf.maximum(0.1 * x, x)


# takes a N-dimensional tensor and a matrix. multiplies Ijk * kl to get Ijl
def broadcast_matmul(A_ijk, B_kl):
    return tf.tensordot(A_ijk, B_kl, [[-1], [-2]])


# helper function for applying the approx_nn_module
# all it really does is strips data from the candidates and reshapes the neighbors
def get_neighbors(candidates, queries, num_neighbors, dimensions):
    approx_nn_module = tf.load_op_library('./ApproxNNOp/src/ApproxNNOp.so')
    with tf.device('/cpu:0'):
        candidate_coordinates = candidates[:, :, :dimensions]
        nn_op = lambda x: approx_nn_module.approx_nn(candidates=x[0], queries=x[1], num_neighbors=num_neighbors,
                                                     dimensions=dimensions)
        nearest_neighbors = tf.map_fn(nn_op, (candidate_coordinates, queries), dtype=tf.int32)
    return nearest_neighbors


def flann_neighbors(candidates, queries, num_neighbors):
    flann = FLANN()
    if candidates.shape[0] < num_neighbors:
        print(candidates.shape, num_neighbors)
    result, distances = flann.nn(candidates, queries, num_neighbors)
    if num_neighbors == 1:
        result = np.expand_dims(result(0))
        distances = np.expand_dims(distances, 0)
    distances = np.expand_dims(distances, -1)
    return result, distances


# NOT BATCHED
def py_get_neighbors(candidates, queries, num_neighbors):
    with tf.name_scope('pyGetNeighbors') as scope:
        candidate_coordinates = candidates
        nearest_neighbors, distances = tf.py_func(flann_neighbors, [candidate_coordinates, queries, num_neighbors],
                                                 [tf.int32, tf.float32])
        return nearest_neighbors, distances


def extract_delta_xyz(queries, candidates, dimensions, neighbors):
    num_neighbors = tf.shape(neighbors)[-1]
    mapped_candidates = tf.map_fn(lambda x: tf.gather(x[0], x[1]), (candidates, neighbors), dtype=tf.float32)
    mapped_candidates = tf.reshape(mapped_candidates, [tf.shape(queries)[0], tf.shape(queries)[1], num_neighbors,
                                                       tf.shape(candidates)[2]])
    spatial_features = mapped_candidates[:, :, :, :dimensions] - tf.expand_dims(queries, 2)
    return spatial_features


def cat_spatial_and_non_spatial(spatial, non_spatial, neighbors):
    non_spatial = tf.expand_dims(non_spatial, -1)
    num_neighbors = tf.shape(neighbors)[-1]
    mapped_non_spatial = tf.map_fn(lambda x: tf.gather(x[0], x[1]), (non_spatial, neighbors), dtype=tf.float32)
    mapped_non_spatial = tf.reshape(mapped_non_spatial, [tf.shape(spatial)[0], tf.shape(spatial)[1], num_neighbors,
                                                         tf.shape(non_spatial)[2]])
    return tf.concat([spatial, mapped_non_spatial], -1)


# DOES NOT APPLY ACTIVATION FOR YOU (but does apply activation function inside)
# because of the peculiarity of each query getting different information
# from the same point (because we are looking at relative distance)
# you have the issue of not being able to apply a 1x1 convolution prior
# to doing the actual general convolution. You have to apply it independently
# for each candidate for each query.
# this means that W now needs to be a LIST of layers so we can apply 1x1
# convolutions (which this desperately needs)
class GeneralizedConvolution(object):
    def __init__(self, W, B, features, keep_prob=None):

        # this is required for 1x1 convolutions to be possible (regrettably)
        # it does the operation for each weight in the set of weights
        # this is essentially a small MLP for each candidate for each query.
        # once again unfortunately it needs to apply activations inside too.
        # (since 1x1 convolutions generally have activations applied after)
        activations = features
        for w, b in zip(W, B)[:-1]:
            self.multed = broadcast_matmul(activations, w) + b
            if keep_prob != None:
                self.multed = tf.nn.dropout(self.multed, keep_prob, noise_shape=tf.shape(self.multed)[-1:])
            normalized = self.multed
            activations = lrelu(normalized)
        self.multed = broadcast_matmul(activations, W[-1]) + B[-1]
        self.linear_activation = tf.reduce_mean(self.multed, axis=2)


# the first layer of layerSizes needs to match the size of the input
# the total size of each point in the output is layerSizes[-1] + 3
# because it also adds the spatial dimensions of the query
# supports batching
class GeneralizedConvolutionLayer(object):
    def __init__(self, layer_sizes, features, keep_prob=None):
        self.W = [weight_variable(shape=[layer_sizes[i], layer_sizes[i + 1]]) for i in range(len(layer_sizes) - 1)]
        self.B = [bias_variable(shape=[layer_sizes[i + 1]]) for i in range(len(layer_sizes) - 1)]
        self.generalized_convolution = GeneralizedConvolution(self.W, self.B, features, keep_prob)
        h = self.generalized_convolution.linear_activation
        self.h_norm = h


# input_array [points, features]
def random_sample_array(input_array, proportion):
    input_total = tf.shape(input_array)[0]
    indices = tf.range(input_total)
    shuffled_indices = tf.random_shuffle(indices)

    output = tf.gather(input_array, shuffled_indices[:tf.cast(proportion * tf.cast(input_total, tf.float32), tf.int32)])
    return output


# only rotates around the z axis
def rotate_cloud(cloud, t):
    rotation_matrix = tf.stack([tf.stack([tf.cos(t), -tf.sin(t), 0.0]),
                                tf.stack([tf.sin(t), tf.cos(t), 0.0]),
                                tf.stack([0.0, 0.0, 1.0])])
    return tf.transpose(tf.matmul(rotation_matrix, tf.transpose(cloud)))
