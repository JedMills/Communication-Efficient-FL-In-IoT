"""
Utils for loading/handling datasets for use with FL.
"""
import numpy as np
import tensorflow as tf


def shuffle_client_data(data):
    """ Returns a copy of a client's (x, y) shuffled using the same order. """
    order = np.random.permutation(data[0].shape[0])
    x = data[0][order]
    y = data[1][order]
    return x, y


def to_oh(data, n_c):
    """
    Converts an array of class numbers into one-hot encoding.

    Parameters:
    data (array):   y values as integers
    n_c (int):      number of classes in data

    Returns:
    oh (array):     y values as one-hot, shape is [len(y), n_c]
    """
    o_h = np.zeros([data.shape[0], n_c], dtype=np.float32)
    o_h[np.arange(data.shape[0]), data] = 1.0
    return o_h


def get_client_data(data, idx):
    """
    Returns (x, y) values using client's index from all clients' data.

    Parameters:
    data (list):    all clients' data (x, y)
    idxs (int):     client index

    Returns:
    (client's x data, client's y (label) data)
    """
    return data[0][idx], data[1][idx]
    
    
    
def split_iid(x, y, W):
    """
    Shuffles x and y training data, and splits shuffled data into W equal parts.
    
    Parameters:
    x (array):  x data of shape [n_samples, sample_shape]
    y (array):  label data of shape [n_samples]
    W (int):    number of portions (Workers) to split data into
    
    Returns:
    xs (list[array]):   worker x data after shuffling, length W 
    ys (list[array]):   worker y data after shuffling, length W
    """
    ord = np.random.permutation(y.shape[0])
    xs = np.array_split(x[ord], W)
    ys = np.array_split(y[ord], W)
    
    return xs, ys
    
    
def split_niid(x, y, W, n_frag):
    """
    Sorts x and y training data by label (y), splits the data into W*n_frag 
    shards, assigns n_frag shard to each worker, and returns data as lists of 
    length W. Although not the most best to split datasets, it is the same 
    as used by McMahan et al in the FedAvg paper.
    
    Parameters:
    x (array):      x data of shape [n_samples, sample_shape]
    y (array):      label data of shape [n_samples]
    W (int):        number of portions (Workers) to split data into
    n_frag (int):   number of fragments/shards per worker
    
    Returns:
    xs (list[array]):   worker x data after non-iid split, length W
    ys (list[array]):   worker y data after non-iid split, length W
    """
    ord = np.argsort(y) # sort by label 
    x_frags = np.array_split(x[ord], W * n_frag)
    y_frags = np.array_split(y[ord], W * n_frag)
        
    # fragment assignments for each worker
    frag_idxs = [[i+(W*j) for j in range(n_frag)] for i in range(W)]
    
    # concatenate fragments for each worker
    xs = []
    ys = []
    for w in range(W):
        xs.append(np.squeeze(np.concatenate([x_frags[i] for i in frag_idxs[w]])))
        ys.append(np.squeeze(np.concatenate([y_frags[i] for i in frag_idxs[w]])))
        
    return xs, ys


def load_dataset(dataset, W, iid):
    """
    Load given dataset, split into W partitions.
    
    Parameters:
    dataset (str):  'mnist' or 'cifar'
    W (int):        number of workers to split training data into
    iid (bool):     shuffle iid or partition non-iid
    
    Returns:
    train (tuple):  (x, y) data, each a list of length W of workers' data
    test (tuple):   (x, y) data, each an array of all test samples/labels
    """
    if dataset == 'cifar':
        # train, test = load_cifar_dataset(folder, W, iid)
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    elif dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = x_train.reshape((-1, 784))
        x_test = x_test.reshape((-1, 784))
    else:
        raise RuntimeError('Unsupported dataset string...')
        
    x_train = (x_train / 255).astype(np.float32)
    x_test = (x_test / 255).astype(np.float32)
        
    if iid:
        x_trains, y_trains = split_iid(x_train, y_train, W)
    else:
        x_trains, y_trains = split_niid(x_train, y_train, W, 2)
        
    y_trains = [to_oh(y, 10) for y in y_trains]
    y_test = to_oh(y_test, 10)
    
    return (x_trains, y_trains), (x_test, y_test)