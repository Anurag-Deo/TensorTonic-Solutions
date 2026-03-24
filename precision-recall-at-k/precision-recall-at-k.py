def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    recommended = recommended[:k]
    intersection = set(recommended).intersection(set(relevant))
    return [len(intersection)/k , len(intersection)/len(relevant)]