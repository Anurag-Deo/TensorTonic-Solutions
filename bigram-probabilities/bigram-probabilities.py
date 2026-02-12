from collections import defaultdict
def bigram_probabilities(tokens):
    """
    Returns: (counts, probs)
      counts: dict mapping (w1, w2) -> integer count
      probs: dict mapping (w1, w2) -> float P(w2 | w1) with add-1 smoothing
    """
    # Your code here
    vocab = set(tokens)
    bigrams = defaultdict(int)
    for i in range(0,len(tokens)-1):
      pair = (tokens[i], tokens[i+1])
      bigrams[pair] += 1
    probs = defaultdict(int)
    for i in vocab:
      for j in vocab:
        item = (i,j)
        probs[item] = (bigrams[item]+1)/(tokens[0:len(tokens)-1].count(item[0])+ len(vocab))
    return bigrams,probs