import numpy as np

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import glob
from random import shuffle


def putFilter(w, f, x, y, z):
    positive = np.log(np.clip(f, 0.0, f.max())**2)
    negative = np.log(np.clip(-f, 0.0, -f.min())**2)
    img = np.empty(f.shape + (4,), dtype=np.ubyte)
    img[..., 0] = negative * (255./negative.max())
    img[..., 1] = positive * (255./positive.max())
    img[..., 2] = img[..., 1]
    img[..., 3] = img[..., 1]*0.3 + img[..., 0]*0.3
    img[..., 3] = (img[..., 3].astype(float)/255.) ** 2 * 255

    v = gl.GLVolumeItem(img)
    v.translate(x,y,z)
    w.addItem(v)


def scale_by_mean(filters):
    mean = np.mean(filters, axis=(0, 1, 2))
    stddev = np.sqrt(np.var(filters, axis=(0, 1, 2)))
    filters = (filters-mean)/stddev
    return filters


def scale_by_range(filters):
    maxes = np.max(filters)
    mins = np.min(filters)
    absMax = np.max((maxes, -mins))
    filters = filters/absMax
    return filters


def draw_filters():
    filters = np.load("../firstLayerFilters.npy")
    filters = scale_by_mean(filters)
    filters = np.rollaxis(filters, -1)

    for i, f in enumerate(filters):
        print(np.mean(f), np.var(f), np.min(f), np.max(f))
        putFilter(w, f, -25+(75*(i % 8-4)), -25+(75*(i/8-4)), -25)


def put_point_cloud(w, cloud, posX, posY):
    points = cloud[:, :3]  # actual xyz
    print(points.shape)
    plot = gl.GLScatterPlotItem(pos=points, color=(255, 0, 0, 255), size=2)
    plot.translate(posX, posY, 0)
    w.addItem(plot)
    pass


def draw_activation_clouds(w):
    files = list()
    for f in glob.glob('../dataSets/ModelNet10csv/*/test/*'):
        files.append(f)
    shuffle(files)
    for i, f in enumerate(files):
        if i == 16:
            break
        cloud = np.loadtxt(f, delimiter=' ')
        put_point_cloud(w, cloud, (i-8)*10, 0)


if __name__ == '__main__':
    app = QtGui.QApplication([])
    w = gl.GLViewWidget()
    w.opts['distance'] = 200
    w.show()
    w.setWindowTitle('pyqtgraph example: GLVolumeItem')
    draw_activation_clouds(w)
    g = gl.GLGridItem()
    g.scale(100, 100, 1)
    w.addItem(g)
    ax = gl.GLAxisItem()
    w.addItem(ax)
    QtGui.QApplication.instance().exec_()
