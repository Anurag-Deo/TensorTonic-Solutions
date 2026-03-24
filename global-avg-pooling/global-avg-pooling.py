import numpy as np

def global_avg_pool(x):
    """
    Compute global average pooling over spatial dims.
    Supports (C,H,W) => (C,) and (N,C,H,W) => (N,C).
    """
    # Write code here
    x = np.array(x, dtype=float)
    if x.ndim not in (3, 4):
        raise ValueError("Input must be 3D or 4D")
    h = x.shape[-2]
    w = x.shape[-1]
    return (1/(h*w))*np.sum(np.sum(x, axis=-1), axis=-1)