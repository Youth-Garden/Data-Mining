from typing import Dict, Tuple
import numpy as np
from sklearn.metrics import f1_score, adjusted_rand_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment


def best_map(y_true: np.ndarray, y_pred: np.ndarray):
    labels_true = np.unique(y_true)
    labels_pred = np.unique(y_pred)
    cost = np.zeros((len(labels_pred), len(labels_true)), dtype=int)
    for i, lp in enumerate(labels_pred):
        for j, lt in enumerate(labels_true):
            cost[i, j] = np.sum((y_pred == lp) & (y_true == lt))
    r, c = linear_sum_assignment(cost.max() - cost)
    mapping = {labels_pred[ri]: labels_true[ci] for ri, ci in zip(r, c)}
    y_mapped = np.array([mapping.get(lp, -1) for lp in y_pred])
    return y_mapped, mapping


def evaluate_clustering(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    y_mapped, mapping = best_map(y_true, y_pred)
    # Ensure JSON-serializable types (avoid numpy.int64 keys/values)
    mapping_serializable = {int(k): int(v) for k, v in mapping.items()}
    return {
        "F1_macro": float(f1_score(y_true, y_mapped, average="macro")),
        "ARI": float(adjusted_rand_score(y_true, y_pred)),
        "NMI": float(normalized_mutual_info_score(y_true, y_pred)),
        "Mapping": mapping_serializable,
    }
