# Differing input pipelines for vectorization and training.

import glob
from generalized_conv import *


def make_and_open(filename, mode):
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    return open(filename, mode)


def get_modelnet10_files(path, subfolder):
    files = list()
    for f in glob.glob(path + '/*/' + subfolder + '/*'):
        files.append(f)
    return files 


def get_modelmet40_files(subfolder):
    files = list()
    for f in glob.glob('dataSets/ModelNet40csv/*/' + subfolder + '/*'):
        files.append(f)
    return files 


def get_mnet10_cloud(filename):
    for k in labels:
        if k in filename:
            label = labels[k]
    cloud = np.loadtxt(filename)
    cloud = np.insert(cloud, 3, 1, axis=1)
    return cloud, label


labels = {
        'bathtub' : 0,
        'bed'     : 1,
        'chair'   : 2,
        'desk'    : 3,
        'dresser' : 4,
        'monitor' : 5,
        'night_stand' : 6,
        'sofa'    : 7,
        'table'   : 8,
        'toilet'  : 9
        }

labels40 = {
        'airplane' 	 : 0,
        'bathtub' 	 : 1,
        'bed'		 : 2,
        'bench'		 : 3,
        'bookshelf'	 : 4,
        'bottle'	 : 5,
        'bowl'		 : 6,
        'car'		 : 7,
        'chair'		 : 8,
        'cone'		 : 9,
        'cup'		 : 10,
        'curtain'	 : 11,
        'desk'		 : 12,
        'door'		 : 13,
        'dresser'	 : 14,
        'f'		     : 15,
        'flower_pot' : 16,
        'glass_box'	 : 17,
        'guitar'	 : 18,
        'keyboard'	 : 19,
        'lamp'		 : 20,
        'laptop'	 : 21,
        'mantel'	 : 22,
        'monitor'	 : 23,
        'night_stand' :24,
        'person'	 : 25,
        'piano'		 : 26,
        'plant'		 : 27,
        'radio'		 : 28,
        'range_hood' : 29,
        'sink'       : 30,
        'sofa'		 : 31,
        'stairs'	 : 32,
        'stool'		 : 33,
        'table'		 : 34,
        'tent'		 : 35,
        'toilet'	 : 36,
        'tv_stand'	 : 37,
        'vase'		 : 38,
        'wardrobe'	 : 39,
        'xbox'		 : 40
        }


# num_eighbors is a list of all the numbers of neighbors except the last layer
def tf_input(filename_queue, pcl_dims, num_neighbors, rotate=False):
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
            serialized_example,
            features={
                'spatialDims': tf.FixedLenFeature([], tf.int64),
                'totalDims': tf.FixedLenFeature([], tf.int64),
                'numPoints': tf.FixedLenFeature([], tf.int64),
                'pclRaw' : tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64),
                'fileName' : tf.FixedLenFeature([], tf.string)
                }
            )
    cloud_raw = tf.decode_raw(features['pclRaw'], tf.float32)
    file_name = tf.cast(features['fileName'], tf.string)
    num_points = tf.cast(features['numPoints'], tf.int32)
    label_raw = tf.cast(features['label'], tf.int32)
    
    cloud_shape = tf.stack([num_points,4])
    cloud = tf.reshape(cloud_raw,cloud_shape)

    tmp_cloud = cloud[:, :pcl_dims]
    
    if rotate:
        features = cloud[:, pcl_dims:]

        # FLIPPING CLOUD
        flip = tf.random_uniform(2, 0, 1, dtype=tf.int32)
        flip = tf.concat((flip, [1]))  # don't flip the z axis
        flip = flip*2 - 1  # should be -1 or 1 now
        tmp_cloud = tmp_cloud * flip
        cloud = tf.concat((tmp_cloud, features), -1)

    q1 = random_sample_array(tmp_cloud, 0.5)
    q2 = random_sample_array(q1, 0.5)
    q3 = random_sample_array(q2, 0.5)
    q4 = tf.zeros([1, pcl_dims], dtype=tf.float32)
    candidates = [tmp_cloud, q1, q2, q3]
    queries = [q1, q2, q3, q4]
    neighbors = list()
    distances = list()
    neighbor_quantities = [num_neighbors, num_neighbors, num_neighbors, tf.shape(q3)[0]]
    
    for c, q, n in zip(candidates, queries, neighbor_quantities):
        tmp = py_get_neighbors(c, q, n)
        neighbors.append(tmp[0])
        distances.append(tmp[1])

    label = tf.one_hot(label_raw, 10)
    label = tf.expand_dims(label, -2)  # expand this since there's only one label for the whole cloud
    return cloud, candidates, queries, label, neighbors, distances, file_name


def batch_input(filenames, batch_size, pcl_dims, num_neighbors, num_epochs=None):
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle = True)
    stuff = tf_input(filename_queue, pcl_dims, num_neighbors)
    min_after_dequeue = 100
    capacity = min_after_dequeue + 3 * batch_size
    return tf.train.shuffle_batch(stuff, batch_size,capacity=capacity, min_after_dequeue=min_after_dequeue)
