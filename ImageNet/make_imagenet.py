# Author: Francesco Visin
# Credits: Yann Dauphin, Francesco Visin
# Licence: BSD 3-clause

"""
Process the uncompressed Imagenet dataset.

The images are resized to 256x256x3. The images are batched in pickle files.
We group the images in pickle files in an effort to reduce the strain on the
filesystem (avoid wasting inodes). Each pickle file contains a list of tuple.
The first element is the JPEG encoded image in a binary string and the second
is the index of the class.
"""
import os
import os.path
import time
from PIL import Image
from tables import Filters, openFile, UInt16Atom, UInt8Atom

import numpy
from scipy.io import loadmat

from dataset_stats import online_stats


def process(path):
    """
    Resize the image to 256x256.

    The smallest dimension is resized to 256x256 and center croping is used for
    the other dimension to reduce to 256.
    """
    im = Image.open(path).convert('RGB')

    im.thumbnail((256 if im.size[0] < im.size[1] else im.size[0],
                  256 if im.size[0] > im.size[1] else im.size[1]),
                 Image.ANTIALIAS)

    if im.size[0] != 256:
        excess = (im.size[0] - 256) / 2
        im = im.crop((excess, 0, 256+excess, 256))
    elif im.size[1] != 256:
        excess = (im.size[1] - 256) / 2
        im = im.crop((0, excess, 256, 256+excess))
    im = numpy.rollaxis(numpy.asarray(im), 2, 0)

    return im


def main(im_path="./val_images",
         meta_file=None,
         gt_file=None,
         out_path="/Tmp/visin/",
         out_name="imagenet_2010",
         is_training_set=False,
         randomize=False,
         complevel=0,
         complib='blosc',
         chunksize=64):
    """This script will process ImageNet images to create an hdf5 dataset.

    Note
    ----
    When processing the training set, the meta.mat file path should be
    provided. When processing either the validation or the test set, the
    ground truth txt file should be provided

    """

    # build the images and labels lists
    # (note: classes starts from 1, so we subtract 1)
    print "{} - Collecting image and labels information ...".format(out_name)
    img_list = []
    if is_training_set:
        assert meta_file is not None
        synsets = loadmat(meta_file)['synsets']
        targets = dict([(synsets[i][0][1].item(), synsets[i][0][0].item() - 1)
                       for i in range(len(synsets))])
        assert min(targets.values()) == 0

        val_targets = []
        for directory, _, images in os.walk(im_path):
            target = targets[directory]
            for img in images:
                im_path = os.path.join(directory, img)
                img_list.append(im_path)
                val_targets.append(target)
    else:
        assert gt_file is not None
        #val_targets = numpy.asarray(map(int, open(gt_file).read().split())) - 1
        val_targets = []
        for directory, _, images in os.walk(im_path):
            for img in images:
                im_path = os.path.join(directory, img)
                img_list.append(im_path)
                val_targets.append(0)
        img_list = sorted(img_list)

    num_img = len(img_list)
    val = zip(img_list, val_targets)

    if randomize:
        numpy.random.seed(0xbeef)
        numpy.random.shuffle(img_list)

    # prepare output file
    print "{} - Preparing the output file ...".format(out_name)
    chunkshape_x = (chunksize, 3, 256, 256)
    chunkshape_y = (chunksize, 1)

    filters = Filters(complevel=complevel, complib=complib, shuffle=True)
    if out_path[-1] != '/':
        out_path += '/'

    out_filename = (out_path + out_name +
                    ('' if complevel == 0 else ('_' + complib + '_')) +
                    ('' if complevel == 0 else str(complevel)) + '.h5')

    if os.path.isfile(out_filename):
        raise ValueError('File {} already exists. Please remove it and launch '
                         'the script again.'.format(out_filename))
    f = openFile(out_filename, mode='w')
    atom8 = UInt8Atom()
    atom16 = UInt16Atom()
    shape_x = (num_img, 3, 256, 256)
    shape_y = (num_img, 1)
    print_freq = 50

    ca_x = f.createCArray(f.root, 'x', atom8, shape_x, filters=filters,
                          chunkshape=chunkshape_x)
    ca_y = f.createCArray(f.root, 'y', atom16, shape_y, filters=filters,
                          chunkshape=chunkshape_y)

    # process the images
    print "{} - Processing files ...".format(out_name)
    begin = time.time()
    i = 0
    for key, target in val:
        i += 1
        if i % print_freq == 0:
            end = time.time()
            print "Processed %d/%d in %.2fs" % (i, len(val), end-begin)
            begin = time.time()
        x, y = (process(key), target)
        ca_x[i-1, ...] = x
        ca_y[i-1, 0] = y

    # collect stats and add them to the hdf5 file
    print "{} - Collecting image stats ...".format(out_name)
    mean, std = online_stats(f.root.x)
    f.createArray(f.root, 'x_mean', mean)
    f.createArray(f.root, 'x_std_dev', std)

    f.close()


if __name__ == "__main__":
    main(im_path="./val_images",
         gt_file="./ILSVRC2010_validation_ground_truth.txt",
         out_path="/Tmp/visin/",
         out_name="imagenet_2010_valid_asdasda",
         is_training_set=False,
         randomize=False)
