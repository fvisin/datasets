# Author: Kyle Kaster
# License: BSD 3-clause
import numpy as np
import tables

def online_stats(X):
    """
    Converted from John D. Cook
    http://www.johndcook.com/blog/standard_deviation/
    """
    prev_mean = None
    prev_var = None
    n_seen = 0.
    for i in range(len(X)):
        if i % 10000 == 0:
            print("Processing image %i of %i" % (i, len(X)))
        n_seen += 1
        X_i = X[i].ravel()
        if prev_mean is None:
            prev_mean = X_i.astype('float32')
            prev_var = 0.
        else:
            curr_mean = prev_mean + (X_i - prev_mean) / n_seen
            curr_var = prev_var + (X_i - prev_mean) * (
                X_i - curr_mean)
            prev_mean = curr_mean
            prev_var = curr_var
    # n - 1 for sample variance, but numpy default is n
    return prev_mean, np.sqrt(prev_var / n_seen)

if __name__ == "__main__":
    f = tables.openFile('imagenet_2010_train.h5')
    data = f.root.x
    m, s = online_stats(data)
    np.savez('imagenet_train_stats.npz', mean=m, std=s)
