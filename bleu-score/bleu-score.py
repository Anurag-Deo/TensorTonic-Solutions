import math
from collections import Counter
def bleu_score(candidate, reference, max_n):
    """
    Compute the BLEU score for a candidate translation.
    """
    # Write code here
    if len(candidate)==0:
        return 0.0
    
    precision = []
    for n in range(1,max_n+1):
        cand_ngrams = Counter(
            tuple(candidate[i:i+n])
            for i in range(len(candidate)-n+1)
        )
        ref_ngram = Counter(
            tuple(reference[i:i+n])
            for i in range(len(reference)-n+1)
        )
        overlap = 0
        for ng in cand_ngrams:
            overlap += min(cand_ngrams[ng], ref_ngram.get(ng,0))
        
        total = sum(cand_ngrams.values())
        if total ==0 :
            precision.append(0)
        else:
            precision.append(overlap/total)
    
    if min(precision) == 0:
        return 0.0
    
    log_sum = sum(math.log(p) for p in precision) / max_n
    geo_mean = math.exp(log_sum)

    c = len(candidate)
    r = len(reference)

    if c > r:
        bp = 1
    else:
        bp = math.exp(1 - r / c)

    return bp * geo_mean

    