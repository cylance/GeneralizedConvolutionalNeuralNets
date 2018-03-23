# Primary script for training generalized convolutional neural networks.

from __future__ import print_function
import time

from models import *
import input_pipelines


POINT_CLOUD_DIMENSIONS = 3
NUM_NEIGHBORS = 10
BATCH_SIZE = 1

if __name__ == '__main__':
    print('starting main function')
    variant = '1000'
    with tf.name_scope('trainPipe') as scope:
        train_queue = tf.train.string_input_producer(['dataSets/ModelNet10csv' + variant + '/train.tfrecords'],
                                                     num_epochs=400)
        train_stuff = input_pipelines.tf_input(train_queue, POINT_CLOUD_DIMENSIONS, NUM_NEIGHBORS, rotate=False)
    with tf.name_scope('testPipe') as scope:
        test_queue = tf.train.string_input_producer(['dataSets/ModelNet10csv' + variant + '/test.tfrecords'],
                                                    num_epochs=400)
        test_stuff = input_pipelines.tf_input(test_queue, POINT_CLOUD_DIMENSIONS, NUM_NEIGHBORS)
    
    is_testing = tf.placeholder_with_default(input=False, shape=())
    keep_prob = tf.placeholder_with_default(input=1.0, shape=())
    stuff = [tf.cond(is_testing, lambda:tes, lambda:trs) for (tes, trs) in zip(test_stuff, train_stuff)]
    
    # expand the dimensions of tensors in nested lists
    expanded_stuff = [[tf.expand_dims(t, 0) for t in s] if isinstance(s, list) else tf.expand_dims(s, 0) for s in stuff]
    cloud, candidates, all_queries, label, all_neighbors, all_distances, filenames = expanded_stuff
    model = GeneralizedConvolutionModel(cloud, candidates, all_queries, POINT_CLOUD_DIMENSIONS, all_neighbors,
                                        all_distances, keep_prob, variant)
    h_final_conv = model.h_final_conv

    # ---------TRAINING----------------
    with tf.name_scope('entropyAndTrainingAndEvaluation') as scope:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        weights = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if 'weight' in v.name]
        for w in weights:
            print(w.name)
        l2_losses = [tf.nn.l2_loss(w) for w in weights]
        l2_loss = 0 * tf.add_n(l2_losses)/len(l2_losses)
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(label * tf.log(h_final_conv + 1e-8), axis=[2]))

        global_step = tf.Variable(0, trainable=False)
        start_learn_rate = 0.005
        learning_rate = tf.train.exponential_decay(start_learn_rate, global_step, 500, 0.96, staircase=True)
        learning_rate = tf.placeholder_with_default(learning_rate, None)

        with tf.control_dependencies(update_ops):
            train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy + l2_loss,
                                                                        global_step=global_step)

    # ---------PERFORMANCE-------------
        argmax_L = tf.argmax(label, 2)
        argmax_conv = tf.argmax(h_final_conv, 2)
        got_correct = tf.equal(tf.argmax(label, 2), tf.argmax(h_final_conv, 2))

    # ---------PERFORMANCE-------------
    print('done building graph')
    with tf.Session() as sess:
        writer = tf.summary.FileWriter('./summaries', sess.graph)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        entropies = list()
        corrects = list()
        train_entropies = list()
        train_corrects = list()
        last_time = time.time()
        
        epochs = 20
        for i in range(epochs*4000):
            if i < 4000 + 1:
                _, ent, correct, max_conv, maxL, rate = sess.run([train_step, cross_entropy, got_correct, argmax_conv,
                                                                  argmax_L, learning_rate],
                                                                 feed_dict={learning_rate: 0.0005})
            else:
                _, ent, correct, max_conv, maxL, rate = sess.run([train_step, cross_entropy, got_correct, argmax_conv,
                                                                  argmax_L, learning_rate])

            entropies.append(ent)
            corrects.extend(correct)  # extend because correct is an array
            if i % 400 == 0:
                last_time = time.time()
                print(rate, np.mean(entropies), np.mean(corrects))
                train_entropies.append(np.mean(entropies))
                train_corrects.append(np.mean(corrects))
                entropies = list()
                corrects = list()
            if i % 4000 == 0:
                print("EPOCH: ", i / 4000, " TRAIN entropy: ", np.mean(train_entropies),
                      " correct: ", np.mean(train_corrects))
                train_entropies = list()
                train_corrects = list()

                print('saving visualization')
                                
                xyz_values = np.meshgrid(np.linspace(-0.6, 0.6, 50), np.linspace(-0.6, 0.6, 50),
                                         np.linspace(-0.6, 0.6, 50))
                flat_xyz_values = [v.flatten() for v in xyz_values]
                xyz_features = np.stack(flat_xyz_values, 1)
                _, distance_feats = flann_neighbors(xyz_features, np.zeros((1, 3)), 125000)
                xyz_features = np.expand_dims(xyz_features, 0)
                final_features = np.concatenate((xyz_features, distance_feats, np.ones((1, 125000, 1))), -1)
                final_features = np.expand_dims(final_features, 0)

                internal_conv = sess.run(model.gConv1.generalized_convolution.multed, feed_dict={model.features_conv1: final_features})
                internal_conv = np.reshape(internal_conv, (50, 50, 50, 16))
                np.save("firstLayerFilters", internal_conv)

                print('running testing')
                weights = sess.run([model.fConv.W[0]])
                np.set_printoptions(precision=4)
                for w in weights:  
                    print(np.mean(w[:5, :]), np.var(w[:5, :]))
                    print(np.mean(w[5:, :]), np.var(w[5:, :]))
                    print()
                t_entropies = list()
                t_corrects = list()
                confusionMatrix = np.zeros((10, 10))

                for i in range(908):
                    ent, correct, fNames, maxL, maxC, conv1 = sess.run(
                            [cross_entropy, got_correct, filenames, argmax_L, argmax_conv, model.h_conv1_cloud],
                            feed_dict={is_testing: True})
                    confusionMatrix[np.squeeze(maxC, 0), np.squeeze(maxL, 0)] += 1

                    t_entropies.append(ent)
                    t_corrects.append(correct)

                print(confusionMatrix)
                print('trying to get mean')
                print(np.mean(t_entropies), np.mean(t_corrects))
                print("starting training again")

        coord.request_stop()
        coord.join(threads)
