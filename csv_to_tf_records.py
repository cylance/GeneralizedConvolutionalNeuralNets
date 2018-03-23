from random import shuffle
import tensorflow as tf
import input_pipelines
import numpy as np
import sys

numNeighbors = 256


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


if __name__=="__main__":
    path = sys.argv[1]
    subfolder = sys.argv[2]
    
    fileNames = input_pipelines.get_modelnet10_files(path, subfolder)
    shuffle(fileNames)

    writer = tf.python_io.TFRecordWriter(path+subfolder+'.tfrecords')

    for i, fName in enumerate(fileNames):
        print(i)
        cloud, label = input_pipelines.get_mnet10_cloud(fName)
        numPoints = cloud.shape[0]
        
        if numPoints>=numNeighbors: 
            # normalize mean for each dimension
            # but normalize over all dimension for variance
            # to avoid squeezing the cloud
            cloud = cloud.astype(np.float32) 
            meanXYZ = np.mean(cloud[:, :3], 0)
            cloud[:, :3] = cloud[:, :3] - meanXYZ
            varXYZ = np.var(cloud[:, :3])
            cloud[:, :3] = cloud[:, :3]/np.sqrt(varXYZ)

            clRaw = cloud.tostring() 

            example = tf.train.Example(features=tf.train.Features(feature={
                'spatialDims': _int64_feature(3),
                'totalDims': _int64_feature(4),
                'numPoints': _int64_feature(numPoints),
                'pclRaw': _bytes_feature(clRaw),
                'label': _int64_feature(label),
                'fileName': _bytes_feature(fName)
                }))
            writer.write(example.SerializeToString())
        else:
            print fName, numPoints

    writer.close()
