import numpy as np
from numpy.random import default_rng
from math import sqrt

def get_vector(seed = 1, upper_bound = 10, n_samples = 10):
    generator = default_rng(seed)
    vector = upper_bound * generator.random((n_samples, 1))  # refer docs; [(upper - lower) * rng.random() + lower] gives range[lower, upper)
    return vector

def and_gate_dataset(positive_samples = 100, seed = 1000):
    generator = default_rng(seed)
    seeds = generator.choice(1000, 4, replace = False)
    # features
    both_positive = np.hstack((get_vector(seed = seeds[0], n_samples = positive_samples), get_vector(seed = seeds[1], n_samples = positive_samples)))
    one_zero_1 = np.hstack((np.zeros((positive_samples//2, 1)), get_vector(seed = seeds[2], n_samples = positive_samples//2)))
    one_zero_2 = np.hstack((get_vector(seed = seeds[3], n_samples = positive_samples//2), np.zeros((positive_samples//2, 1))))
    both_zero = np.array([0, 0])
    features = np.vstack((both_positive, one_zero_1, one_zero_2, both_zero))

    # labels
    labels_positive = np.ones((positive_samples, 1), dtype = int)
    labels_negative = np.zeros((positive_samples//2 + positive_samples//2 + 1, 1), dtype = int)
    labels = np.vstack((labels_positive, labels_negative))

    dataset = np.hstack((features, labels))
    generator.shuffle(dataset)
    return dataset[:, [0, 1]], dataset[:, [2]]

def get_minibatch(features, targets, batch_size = 1, start_at = 0):
    # give this the same stuff every time
    if start_at >= features.shape[0]:
        print("invalid start_at for get_minibatch")
        return None, None
    return features[start_at:min(features.shape[0], start_at + batch_size)], targets[start_at:min(targets.shape[0], start_at + batch_size)]

@np.vectorize(excluded = {1, 2, "mean", "std"})
def make_standard(x, mean, std):
    return (x - mean) / std

def standardize_data(features, from_means = None, from_stds = None):  # in-place operation
    means = list()
    stds = list()
    for i in range(features.shape[1]):
        mean = np.mean(features[:, i]) if from_means is None else from_means[i]
        std = np.std(features[:, i]) if from_stds is None else from_stds[i]
        features[:, i] = (features[:, i] - mean) / std
        means.append(mean)
        stds.append(std)
    return means, stds
    

def linear_regression_dataset(samples = 200, x_start = 0, x_end = 100, slope = 1, intercept = 5, sigma_ = 8, seed = 7):
    x = np.linspace(x_start, x_end, samples)

    y = slope * x + intercept
    y += default_rng(seed).normal(0, sigma_, samples)

    return x.reshape((x.size, 1)), y.reshape((y.size, 1))