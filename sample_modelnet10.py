# Used in vectorization to create point clouds from mesh models.

import glob
import os
import errno
import numpy as np
from joblib import Parallel, delayed
import trimesh
from trimesh import sample
import sys


NEW_PATH = sys.argv[1]
NUM_SAMPLES = int(sys.argv[2])


def volume(points):

    mins = np.min(points, axis=0)
    maxes = np.max(points, axis=0)
    dimensions = maxes-mins
    volume = np.prod(dimensions)
    return volume


def sample_file(filename):
    new_filename = filename.replace('dataSets/ModelNet10/', NEW_PATH)
    new_filename = new_filename.replace('.off', '.csv')
    if not os.path.exists(os.path.dirname(new_filename)):
        try:
            os.makedirs(os.path.dirname(new_filename))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    mesh = trimesh.load_mesh(filename)
    vol = volume(mesh.vertices)
    mesh.vertices = mesh.vertices/np.power(vol, 0.3333333333333333333333)*2
    cloud = sample.sample_surface(mesh, NUM_SAMPLES)
    cloud = cloud[0]
    np.savetxt(new_filename, cloud)
    print(cloud.shape[0], filename, new_filename)

    
if __name__ == '__main__':
    Parallel(n_jobs=7)(delayed(sample_file)(f) for f in glob.glob('dataSets/ModelNet10/*/*/*'))
    

