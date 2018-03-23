# For gathering statistics used in normalizing xyz features.

import generalized_conv
import numpy as np
import input_pipelines
import sys
from joblib import Parallel, delayed


num_neighbors = int(sys.argv[2])


def get_diffs_and_dists(query, cloud, num_neighbors):
    neighbors, distances = generalized_conv.flann_neighbors(cloud, query, num_neighbors)
    mapped_neighbors = np.take(cloud, neighbors, 0)
    query = np.expand_dims(query, 1)
    differences = mapped_neighbors - query
    return differences, distances


def random_sample_cloud(cloud, proportion):
    num_points = cloud.shape[0]
    indices = np.arange(num_points)
    np.random.shuffle(indices)
    return cloud[indices[:int(num_points*proportion)], :]


def get_stats(i, filename):
    cloud, label = input_pipelines.get_mnet10_cloud(filename)
    cloud = cloud[:, :3]
    cloud = cloud.astype(np.float32)

    stats_list = list()

    query1 = random_sample_cloud(cloud, 0.5)
    stats_list.append(get_diffs_and_dists(query1, cloud, num_neighbors))
    
    query2 = random_sample_cloud(query1, 0.5)
    stats_list.append(get_diffs_and_dists(query2, query1, num_neighbors))

    query3 = random_sample_cloud(query2, 0.5)
    stats_list.append(get_diffs_and_dists(query3, query2, num_neighbors))
    
    final_query = np.zeros([1, 3]).astype(np.float32)
    stats_list.append(get_diffs_and_dists(final_query, query3, query3.shape[0]))
    if i % 10 == 0:
        print(i)
    return stats_list


if __name__ == "__main__":
    path = sys.argv[1]
    fileNames = input_pipelines.get_modelnet10_files(path, 'train')
    fileNames.extend(input_pipelines.get_modelnet10_files(path, 'test'))

    allVals = Parallel(n_jobs=8)(delayed(get_stats)(i, f) for i, f in enumerate(fileNames))
    allVals = np.array(allVals)
    allVals = np.rollaxis(allVals, 0, 3)
    print(allVals.shape)
        
    statistics = list()
    for s in allVals:
        diff = np.stack(s[0, :])
        dist = np.stack(s[1, :])
        diffStat = (np.mean(diff), np.sqrt(np.var(dist)))
        distStat = (np.mean(dist), np.sqrt(np.var(dist)))
        catStat = np.array([diffStat, diffStat, diffStat, distStat])
        print(catStat)
        statistics.append(catStat)

    np.savetxt(path + 'stats1.csv', statistics[0], delimiter=',')
    np.savetxt(path + 'stats2.csv', statistics[1], delimiter=',')
    np.savetxt(path + 'stats3.csv', statistics[2], delimiter=',')
    np.savetxt(path + 'stats4.csv', statistics[2], delimiter=',')




    

    

