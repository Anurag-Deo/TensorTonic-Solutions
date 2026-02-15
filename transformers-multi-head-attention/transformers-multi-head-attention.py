import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    # Your code here
    batch_size, seq_len, d_model = Q.shape
    d_k = d_model//num_heads
    Q_proj = Q@W_q
    K_proj = K@W_k
    V_proj = V@W_v

    def split_heads(x):
      x = x.reshape(batch_size, seq_len, num_heads, d_k)
      return np.transpose(x, (0,2,1,3))

    Q_heads = split_heads(Q_proj)
    K_heads = split_heads(K_proj)
    V_heads = split_heads(V_proj)

    scores = Q_heads @ np.transpose(K_heads, (0,1,3,2))
    scores = scores/np.sqrt(d_model)

    attention_weights = softmax(scores, axis=-1)
    attention_output = attention_weights @ V_heads

    attention_output = np.transpose(attention_output, (0, 2, 1, 3))
    attention_output = attention_output.reshape(batch_size, seq_len, d_model)

    output = attention_output @ W_o

    return output
    