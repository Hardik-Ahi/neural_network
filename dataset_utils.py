import numpy as np
from numpy.random import default_rng
from math import sqrt

def get_vector(seed = 12345, upper_bound = 10, n_samples = 10, zeros = False):
    generator = default_rng(seed)
    vector = None
    if zeros:
        vector = generator.integers(0, 1, size = (n_samples, 1))
    else:
        vector = generator.integers(1, upper_bound, (n_samples, 1))
    return vector

def and_gate_dataset():
    # features
    both_positive = np.hstack((get_vector(seed = 187, n_samples = 100), get_vector(seed = 9, n_samples = 100)))
    one_zero_1 = np.hstack((get_vector(zeros = True, n_samples = 50), get_vector(seed = 45, n_samples = 50)))
    one_zero_2 = np.hstack((get_vector(seed = 987, n_samples = 50), get_vector(zeros = True, n_samples = 50)))
    both_zero = np.array([0, 0])
    features = np.vstack((both_positive, one_zero_1, one_zero_2, both_zero))

    # labels
    labels_positive = np.ones((100, 1), dtype = int)
    labels_negative = np.zeros((101, 1), dtype = int)
    labels = np.vstack((labels_positive, labels_negative))

    '''
    features = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    labels = np.array([[0], [0], [0], [1]])'''

    # dataset
    dataset = np.hstack((features, labels))
    return dataset

def get_minibatch(features, targets, batch_size = 1, start_at = 0):
    # give this the same shuffled stuff every time
    if start_at >= features.shape[0]:
        print("invalid start_at for get_minibatch")
        return None, None
    return features[start_at:min(features.shape[0], start_at + batch_size)], targets[start_at:min(targets.shape[0], start_at + batch_size)]

@np.vectorize(excluded = {1, 2, "mean", "std"})
def make_standard(x, mean, std):
    return (x - mean) / std

def standardize_data(features):
    result = np.array([])
    for i in range(features.shape[1]):
        column = features[:, i]
        mean = 0
        for j in range(column.shape[0]):
            mean += column[j]
        mean = mean / column.shape[0]

        var = 0
        for j in range(column.shape[0]):
            var += ((column[j] - mean)**2)
        var = var / column.shape[0]
        std = sqrt(var)

        if result.size == 0:
            result = make_standard(column, mean, std).reshape((column.shape[0], 1))
        else:
            result = np.hstack((result, make_standard(column, mean, std).reshape((column.shape[0], 1))))
    return result