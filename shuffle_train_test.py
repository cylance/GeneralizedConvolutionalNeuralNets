# For creating your own splits of the train and test data.

from random import shuffle
import tensorflow as tf
import input_pipelines
import numpy as np


numNeighbors = 256


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def convert_list(file_list, subfolder):
    writer = tf.python_io.TFRecordWriter('dataSets/ModelNet10' + subfolder + '.tfrecords')
     
    for i, fName in enumerate(file_list):
        print(i)
        cloud, label = input_pipelines.get_mnet10_cloud(fName)
        num_points = cloud.shape[0]
        
        if num_points >= numNeighbors:
            # normalize mean for each dimension
            # but normalize over all dimesnison for variance
            # to avoid squeezing the cloud
            cloud = cloud.astype(np.float32) 
            mean_xyz = np.mean(cloud[:, :3], 0)
            cloud[:, :3] = cloud[:, :3] - mean_xyz
            var_xyz = np.var(cloud[:, :3])
            cloud[:, :3] = cloud[:, :3]/np.sqrt(var_xyz)

            clRaw = cloud.tostring() 

            example = tf.train.Example(features=tf.train.Features(feature={
                'spatialDims': _int64_feature(3),
                'totalDims': _int64_feature(4),
                'numPoints': _int64_feature(num_points),
                'pclRaw': _bytes_feature(clRaw),
                'label': _int64_feature(label),
                'fileName': _bytes_feature(fName)
                }))
            writer.write(example.SerializeToString())
        else:
            print fName, num_points

    writer.close()


if __name__ == "__main__":
    fileNames = input_pipelines.get_modelnet10_files('train')
    fileNames.extend(input_pipelines.get_modelnet10_files('test'))
    shuffle(fileNames)
    ind = int(len(fileNames)*0.8)
    trainFiles = fileNames[:ind]
    testFiles = fileNames[ind:]
    print(len(testFiles), len(trainFiles))
    convert_list(trainFiles, 'train')
    convert_list(testFiles, 'test')
