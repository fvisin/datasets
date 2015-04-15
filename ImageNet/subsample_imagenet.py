import time
from tables import Filters, openFile, UInt8Atom, UInt16Atom

# options
complevel = 1
complib = 'blosc'
chunksize = 64
chunkshape_x = (chunksize, 3, 256, 256)
chunkshape_y = (chunksize, 1)
# data will be written in the new dataset in blocks of 'batch' size
# does not effect the chunks' size of the new dataset
# batch = [215, 240] --> good to copy all the dataset
batch = [2, 2]
# number of elements to be copied (-1 to copy all of them)
numel = 2

# variables
numel_str = '' if numel == -1 else '_' + str(numel)
filters = Filters(complevel=complevel, complib=complib, shuffle=True)

fr_list = [openFile('/Tmp/visin/imagenet_2010_train_blosc_1.h5', mode='r'),
           openFile('/Tmp/visin/imagenet_2010_test_blosc_1.h5', mode='r')]
fw_list = [openFile('/Tmp/visin/imagenet_2010_train_' + complib + '_' +
                    str(complevel) + numel_str + '.h5', mode='w'),
           openFile('/Tmp/visin/imagenet_2010_test_' + complib + '_' +
                    str(complevel) + numel_str + '.h5', mode='w')]
atom_x = UInt8Atom()
atom_y = UInt16Atom()
shape_x = [f.root.x.shape for f in fr_list]
shape_y = [(f.root.y.shape[0], 1) for f in fr_list]

# create new copy
for fr, fw, b, sh_x, sh_y in zip(fr_list, fw_list, batch, shape_x, shape_y):
    # x
    ca_x = fw.createCArray(fw.root, 'x', atom_x, sh_x, filters=filters,
                           chunkshape=chunkshape_x)
    ca_y = fw.createCArray(fw.root, 'y', atom_y, sh_y, filters=filters,
                           chunkshape=chunkshape_y)
    t0 = time.clock()

    tot_el = sh_x[0] if numel == -1 else numel
    for i in range(tot_el/b):
        if i % 10 == 0:
            print 'Processing batch ' + str(i) + '/' + str(numel/b)
        ca_x[i*b:(i+1)*b, ...] = fr.root.x[i*b:(i+1)*b, ...]
        ca_y[i*b:(i+1)*b, ...] = fr.root.y[i*b:(i+1)*b, ...]
    print ('%.3f seconds to create the dataset' % round(time.clock() - t0, 3))

    fw.close()
    fr.close()
