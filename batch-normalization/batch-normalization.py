import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    """
    Forward-only BatchNorm for (N,D) or (N,C,H,W).
    """
    # Write code here
    x = np.array(x)
    gamma = np.array(gamma)
    beta = np.array(beta)
    if x.ndim == 2:
        mu = np.mean(x, axis=0, keepdims = True)
        sigma = np.var(x, axis=0, keepdims=True)
        
        x_hat = (x-mu)/np.sqrt(sigma+eps)
        return gamma*x_hat + beta
    elif x.ndim == 4:
        mu = np.mean(x, axis=(0,2,3), keepdims=True)
        sigma = np.var(x, axis=(0,2,3), keepdims=True)
        x_hat = (x-mu)/np.sqrt(sigma+eps)
        gamma = gamma.reshape(1,-1,1,1)
        beta = beta.reshape(1,-1,1,1)
        return gamma*x_hat + beta