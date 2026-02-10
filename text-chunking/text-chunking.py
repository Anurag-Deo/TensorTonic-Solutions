def text_chunking(tokens, chunk_size, overlap):
    """
    Split tokens into fixed-size chunks with optional overlap.
    """
    # Write code here
    step = chunk_size-overlap
    ans = []
    for i in range(0,len(tokens),step):
        ans.append(tokens[i:i+chunk_size])
        if i+chunk_size >= len(tokens):
            break
    return ans