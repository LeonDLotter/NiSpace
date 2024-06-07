import numpy as np
from sklearn.model_selection import train_test_split


def _get_dist_dep_splits(dist_mat, train_pct=0.75):
    """
    """
    
    train_test_idc = []
    for i in range(dist_mat.shape[0]):
        distances = dist_mat[i, :]  # for every node
        idx = np.argsort(distances)
        split_idx = int(np.floor(train_pct * len(dist_mat)))
        train_test_idc.append(
            (idx[:split_idx], idx[split_idx:])
        )
        
    return train_test_idc


def _get_rand_splits(n_obs, train_pct=0.75, seed=None):
    """
    """
    idx = np.arange(0, n_obs + 1, 1)
    
    train_test_idc = []
    for i in range(n_obs):
        train_idx, test_idx = train_test_split(
            idx,
            train_size=train_pct,
            random_state=seed+i if seed is not None else None,
        )
        train_test_idc.append(
            (idx[train_idx], idx[test_idx])
        )
        
    return train_test_idc