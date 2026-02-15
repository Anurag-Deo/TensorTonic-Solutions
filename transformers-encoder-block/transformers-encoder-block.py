import numpy as np

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization.
    """
    # Your code here
    mu = np.mean(x, axis=-1, keepdims=True)
    sigm2 = np.var(x, axis=-1, keepdims=True)
    return gamma*((x-mu)/np.sqrt(sigm2+eps)) + beta

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Multi-head attention.
    """
    # Your code here
    batch, seq_len, d_model = Q.shape
    d_k = d_model//num_heads

    Q_proj = Q @ W_q
    K_proj = K @ W_k
    V_proj = V @ W_v

    def split_head(x):
      x = x.reshape(batch, seq_len, num_heads, d_k)
      return np.transpose(x, (0,2,1,3))

    Q_heads = split_head(Q_proj)
    K_heads = split_head(K_proj)
    V_heads = split_head(V_proj)

    score = Q_heads @ np.transpose(K_heads, (0,1,3,2))
    score = score / np.sqrt(d_k)

    attention_weight = softmax(score)
    attention = attention_weight @ V_heads

    attention = np.transpose(attention, (0,2,1,3))
    attention = attention.reshape(batch, seq_len, d_model)

    output = attention @ W_o
    return output

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Position-wise feed-forward network.
    """
    # Your code here
    hidden = x @ W1 + b1
    hidden = np.maximum(0, hidden)
    return hidden @ W2 + b2

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Complete encoder block: MHA + FFN with residuals and layer norms.
    """
    # Your code here
    Q,K,V = x@W_q, x@W_k, x@W_v
    mha = multi_head_attention(x,x,x, W_q, W_k, W_v, W_o, num_heads)
    y = layer_norm(x+mha, gamma1, beta1)
    ff = feed_forward(y, W1, b1, W2, b2)
    return layer_norm(y + ff, gamma2, beta2)