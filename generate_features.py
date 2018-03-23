from generalized_conv import *
import numpy as np

xyzVals = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))
flatXYZVals = [v.flatten() for v in xyzVals]
xyzFeats = np.stack(flatXYZVals, 1)
_, distance_feats = flann_neighbors(xyzFeats, np.zeros((1, 3)), 1000)
xyzFeats = np.expand_dims(xyzFeats, 0)
finalFeats = np.concatenate((xyzFeats,distance_feats, np.ones((1, 1000, 1))), -1)
finalFeats = np.expand_dims(finalFeats, 0)
