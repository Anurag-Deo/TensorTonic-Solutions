import numpy as np

def roc_curve(y_true, y_score):
    """
    Compute ROC curve from binary labels and scores.

    Returns:
        fpr, tpr, thresholds
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    # Sort scores descending
    order = np.argsort(-y_score)
    y_true = y_true[order]
    y_score = y_score[order]

    # Total positives and negatives
    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)

    # Find indices where score changes (unique thresholds)
    distinct_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_indices, len(y_score) - 1]

    # Cumulative TP and FP at those indices
    tps = np.cumsum(y_true == 1)[threshold_idxs]
    fps = np.cumsum(y_true == 0)[threshold_idxs]

    # Convert to rates
    tpr = tps / P if P > 0 else np.zeros_like(tps, dtype=float)
    fpr = fps / N if N > 0 else np.zeros_like(fps, dtype=float)

    # Threshold values
    thresholds = y_score[threshold_idxs]

    # Add starting point (all predicted negative)
    tpr = np.r_[0.0, tpr]
    fpr = np.r_[0.0, fpr]
    thresholds = np.r_[np.inf, thresholds]

    return fpr, tpr, thresholds