import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    # Write code here
    pos = np.arange(seq_len)[:, np.newaxis]
    dims = np.arange(d_model)[np.newaxis,:]

    angle = 1/np.power(base,(2*(dims//2)/d_model))
    rate = pos*angle
    pe = np.zeros((seq_len,d_model))
    pe[:,0::2] = np.sin(rate[:,0::2])
    pe[:,1::2] = np.cos(rate[:,1::2])
    return pe