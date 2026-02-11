import numpy as np
from collections import Counter
import math

def bm25_score(query_tokens, docs, k1=1.2, b=0.75):
    """
    Returns numpy array of BM25 scores for each document.
    """
    # Write code here
    N = len(docs)
    if N==0:
        return np.array([])
    avgdl = sum([len(item) for item in docs])/N

    doc_lens = [len(doc) for doc in docs]

    df = Counter()
    for doc in docs:
        unique_terms = set(doc)
        for term in unique_terms:
            df[term] += 1
    
    scores = np.zeros(N)
    
    for q in query_tokens:
        if q not in df:
            continue
        
        idf = math.log(1 + (N - df[q] + 0.5) / (df[q] + 0.5))
        
        for i, doc in enumerate(docs):
            tf = doc.count(q)
            if tf == 0:
                continue
            
            denom = tf + k1 * (1 - b + b * (doc_lens[i] / avgdl))
            score = idf * ((tf * (k1 + 1)) / denom)
            scores[i] += score
    
    return scores

