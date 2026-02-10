def remove_stopwords(tokens, stopwords):
    """
    Returns: list[str] - tokens with stopwords removed (preserve order)
    """
    # Your code here
    stopwords = set(stopwords)
    final_tokens = []
    for token in tokens:
        if token not in stopwords:
            final_tokens.append(token)
    return final_tokens