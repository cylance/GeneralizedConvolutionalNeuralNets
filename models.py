# The home of the exact model used in experiments.

from generalized_conv import *


class GeneralizedConvolutionModel:
    """
        class wrapper for model creation so we can still access things inside of the class without
        going through collections

    """

    def __init__(self, cloud, candidates, queries, pcl_dim, premade_neighbors, premade_features, keep_prob, variant):
        with tf.name_scope('model') as scope:
            
            spatial_features = list()
            # currently in the form x,y,z,d
            statistics = [np.loadtxt('dataSets/ModelNet10csv'+variant+'/stats1.csv', delimiter=','),
                          np.loadtxt('dataSets/ModelNet10csv'+variant+'/stats2.csv', delimiter=','),
                          np.loadtxt('dataSets/ModelNet10csv'+variant+'/stats3.csv', delimiter=','),
                          np.loadtxt('dataSets/ModelNet10csv'+variant+'/stats4.csv', delimiter=',')]

            for c, q, n, f, s in zip(candidates, queries, premade_neighbors, premade_features, statistics):
                xyzF = extract_delta_xyz(q, c, pcl_dim, n)
                s_feats = tf.concat([xyzF, f], -1)
                s_feats = (s_feats-s[:, 0])/s[:, 1]
                spatial_features.append(s_feats)

            spatial_dim = pcl_dim + 1
            with tf.name_scope('conv1') as scope:
                self.features_conv1 = cat_spatial_and_non_spatial(spatial_features[0], cloud[:, :, 3:],
                                                                  premade_neighbors[0])
                self.features_conv1 = tf.placeholder_with_default(self.features_conv1, [1, None, None, None])
                self.gConv1 = GeneralizedConvolutionLayer([spatial_dim + 1, 32, 16], self.features_conv1)
                self.h_conv1 = lrelu(self.gConv1.h_norm)
                self.h_conv1_cloud = tf.concat((queries[0], self.h_conv1), 2)

            with tf.name_scope('conv2') as scope:
                features_conv2 = cat_spatial_and_non_spatial(spatial_features[1], self.h_conv1, premade_neighbors[1])
                self.gConv2 = GeneralizedConvolutionLayer([spatial_dim + 16, 64, 32], features_conv2, keep_prob)
                self.h_conv2 = lrelu(self.gConv2.h_norm)
                self.h_conv2_cloud = tf.concat((queries[0], self.h_conv2), 2)

            with tf.name_scope('conv3') as scope:
                features_conv3 = cat_spatial_and_non_spatial(spatial_features[2], self.h_conv2, premade_neighbors[2])
                self.gConv3 = GeneralizedConvolutionLayer([spatial_dim + 32, 128, 64], features_conv3)
                self.h_conv3 = lrelu(self.gConv3.h_norm)
                self.h_conv3_cloud = tf.concat((queries[0], self.h_conv3), 2)

            with tf.name_scope('finalConv') as scope:
                normal_features = self.h_conv3
                self.features_final_conv = cat_spatial_and_non_spatial(spatial_features[3], normal_features,
                                                                       premade_neighbors[3])
                self.features_final_conv = tf.placeholder_with_default(self.features_final_conv, [1, None, None, None])
                self.fConv = GeneralizedConvolutionLayer([spatial_dim + 64, 128, 128, 10], self.features_final_conv[:, :, :, :],
                                                         keep_prob=keep_prob)
                self.h_final_conv = tf.nn.softmax(self.fConv.h_norm)
