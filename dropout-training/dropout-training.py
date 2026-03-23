import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    # Write code here
    if rng is None:
        rng = np.random
    x = np.array(x, dtype=float)

    mask = rng.random(x.shape) > p
    output = x*mask / (1-p)

    return output,mask/(1-p)