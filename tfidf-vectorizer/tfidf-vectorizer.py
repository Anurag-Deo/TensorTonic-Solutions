import numpy as np
from collections import Counter
import math

def tfidf_vectorizer(documents):
    """
    Build TF-IDF matrix from a list of text documents.
    Returns tuple of (tfidf_matrix, vocabulary).
    """
    # Write code here
    N = len(documents)
    vocab = set()
    tokenized_docs = []
    for docs in documents:
        tokens = docs.lower().split()
        tokenized_docs.append(tokens)
        vocab.update(tokens)
    vocab = sorted(vocab)
    V=  len(vocab)

    df = Counter()
    for tokens in tokenized_docs:
        unique = set(tokens)
        for token in unique:
            df[token]+=1
    
    idf = {}
    for term in vocab:
        idf[term] = math.log(N/df[term])

    tfidf_matrix = np.zeros((N, V))
    for i,tokens in enumerate(tokenized_docs):
        tf = Counter(tokens)
        doc_len = len(tokens)

        for j, term in enumerate(vocab):
            term_tf = tf[term] / doc_len  # normalized term frequency
            tfidf_matrix[i, j] = term_tf * idf[term]
    
    return tfidf_matrix, vocab


