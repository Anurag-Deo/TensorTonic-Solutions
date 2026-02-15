import numpy as np

def patch_embed(image: np.ndarray, patch_size: int, embed_dim: int) -> np.ndarray:
    """
    Convert image to patch embeddings.
    """
    # YOUR CODE HERE
    b,h,w,c = image.shape
    n = h*w//(patch_size**2)
    img = image.reshape(b, h//patch_size, patch_size, w//patch_size, patch_size, c)
    img = np.transpose(img, (0,1,3,2,4,5))
    img = img.reshape(b, n, patch_size * patch_size * c)
    W = np.random.randn(patch_size * patch_size * c, embed_dim) * 0.02
    return img @ W
    