import numpy as np

def batch_generator(X, y, batch_size, rng=None, drop_last=False):
    """
    Randomly shuffle a dataset and yield mini-batches (X_batch, y_batch).
    """
    ind = np.arange(len(X))
    if not rng:
        rng = np.random
    rng.shuffle(ind)
    for i in range(0,len(X),batch_size):
        split = ind[i:i+batch_size]
        if drop_last and len(split) < batch_size:
            continue
        X_batch = [X[j] for j in split]
        y_batch = [y[j] for j in split]
        yield np.array(X_batch),np.array(y_batch)
        
    