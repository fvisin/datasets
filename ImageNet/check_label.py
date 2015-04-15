import tables

import scipy.io
import numpy as np
from numpy.random import random_integers
from PIL import Image


def createDic():
    f = scipy.io.loadmat('./meta.mat')
    synsets = f['synsets']

    id2lab = dict()
    for ii, ss in enumerate(synsets):
        if ii >= 1000:
            break
        id2lab[ss[0][0][0][0]] = ss[0][3][0]
    return id2lab


def interactiveQuery():
    id2lab = createDic()
    while True:
        inp = raw_input('Enter ID:')
        if int(inp) + 1 not in id2lab:
            print 'Unknown'
            continue
        print id2lab[int(inp) + 1]


def randomSample(f):
    id2lab = createDic()

    f = tables.File(f, 'r')
    x = f.root.x
    y = f.root.y

    num_images = len(x)

    while True:
        i = random_integers(num_images)

        Image.fromarray(np.swapaxes(x[i], 0, 2)).save('sample.jpg')
        if y[i][0] + 1 not in id2lab:
            print 'Unknown'
            continue
        print id2lab[y[i][0] + 1] + '\n'
        raw_input()

if __name__ == "__main__":
    randomSample('/Tmp/visin/imagenet_2010_valid_new.h5')
