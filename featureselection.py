import numpy as np
from sklearn.feature_selection import mutual_info_classif
from ReliefF import ReliefF
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from skfeature.function.information_theoretical_based import FCBF

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
    else:
        return cfs(X_train, y_train, n_features)

def run_subset_selection(selector, X_train, y_train, n_features, estimator):
    if selector == 'sfs':
        return sfs(X_train, y_train, n_features, estimator)


def sfs(X_train, y_train, n_features, estimator): 
    if estimator == 'rdforest' or estimator == 'rdforest_bag' or estimator == 'rdforest_boost':
        algorithm = RandomForestClassifier(n_estimators=300, random_state=0)
    elif estimator == 'svm' or estimator == 'svm_bag':
        algorithm = LinearSVC(C=0.1, random_state=0, max_iter=10000)
    elif estimator == 'knn':
        algorithm = KNeighborsClassifier(n_neighbors=5)
    elif estimator == 'naive_bayes':
        algorithm = GaussianNB()
    sfs1 = SFS(algorithm, k_features=n_features, forward=True, floating=False, scoring='accuracy', cv=0)
    sfs1 = sfs1.fit(X_train, y_train)
    return sfs1.k_feature_idx_