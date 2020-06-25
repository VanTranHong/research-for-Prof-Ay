import numpy as np
from sklearn.feature_selection import mutual_info_classif
from ReliefF import ReliefF
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

def infogain(X, y, n_features):
    """Runs infogain feature selection on the data (X) and the target values (y) and finds
    the index of the top n_features number of features.

    Args:
        X (Numpy Array): An array containing the dataset.
        y (Numpy Array): An array consisting of the target values
        n_features (int): An integer specifying how many features should be selected.

    Returns:
        List: Returns a list containing the indices of the features that have been selected.
    """
    score = mutual_info_classif(X, y,random_state=0)
    index = list(np.argsort(list(score))[-1*n_features:])
    index.sort()
    return index

def reliefF(X, y, n_features):
    """Runs ReliefF algorithm on the data (X) and the target values (y) and finds 
    the index of the top n_features number of features.

    Args:
        X (Numpy Array): An array containing the dataset.
        y (Numpy Array): An array consisting of the target values.
        n_features (int): An integer specifying how many features should be selected.

    Returns:
        List: Returns a list containing the indices of the features that have been selected.
    """
    fs = ReliefF(n_neighbors=100, n_features_to_keep=n_features, random_state=0)
    fs.fit_transform(X, y)
    index = fs.top_features[:n_features]
    return index

def run_feature_selection(method,X,y,n_features):
    """Runs the specific ranking based feature selection method.

    Args:
        method (String): A string that refers to the feature selection method to be used.
        X (Numpy Array): An array containing the dataset.
        y (Numpy Array): An array consisting of the target values.
        n_features (int): An integer specifying how many features should be selected.

    Returns:
        List: Returns a list containing the indices of the features that have been selected.
    """
    if method == 'infogain':
        return infogain(X,y,n_features)
    elif method == 'reliefF':
        return reliefF(X,y,n_features)