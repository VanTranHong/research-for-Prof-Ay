import pandas as pd
import numpy as np
import featureselection as fselect
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import  SGDClassifier


def bag_elasticnet(X_train,X_test,y_train,y_test):
    
    '''
    applying bagging to logistic regression with elasticnet
    Args:
        X_train, X_test, y_train, y_test

    Returns:
        DataFrame: Preprocessed DataFrame. where Alpha and L1 ratio are hyperparameters of elastic net, estimator is hyperparameter for bagging, confusion matrix is the confusion matrix for each combination of those hyperparameters
    '''
    
    df = pd.DataFrame(columns=['Alpha','L1 ratio','Estimators','Confusion Matrix'])
    rows = []
    alphas= [0.0001, 0.001]#, 0.01, 0.1, 1, 10, 100]
    l1s = [0.0,0.1,0.2]#,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    estimators = [50,100,150]

    for al in alphas:
        for l1 in l1s:

            estimator = SGDClassifier(loss = 'log',alpha= al,penalty = 'elasticnet',l1_ratio=l1,random_state=0)
            for n_est in estimators:

                bag = BaggingClassifier(estimator, n_estimators=n_est,  random_state=0)
                bag.fit(X_train,y_train)
                predicted_labels = bag.predict(X_test)
                tn, fp, fn, tp = confusion_matrix(y_test, predicted_labels, labels=[0,1]).ravel()
                convert_matrix = [tn,fp,fn,tp]
                rows.append([n,n_est,convert_matrix])
                model = regr.fit(X_train, y_train)
                predicted_labels = model.predict(X_test)
                tn, fp, fn, tp = confusion_matrix(y_test, predicted_labels, labels=[0,1]).ravel()
                convert_matrix = [tn,fp,fn,tp]
                rows.append([al,l1,n_est,convert_matrix])
    for i in range(len(rows)):
        df = df.append({'Alpha':rows[i][0],'L1 ratio':rows[i][1],'Estimators':rows[i][2],'Confusion Matrix':rows[i][3]}, ignore_index=True)

    return df




def bag_KNN(X_train,X_test,y_train,y_test):
    
        '''
    applying bagging to logistic regression with elasticnet
    Args:
        X_train, X_test, y_train, y_test

    Returns:
        DataFrame: Preprocessed DataFrame. where Neighbors is hyperparameters of KNN, estimator is hyperparameter for bagging, confusion matrix is the confusion matrix for each combination of those hyperparameters
    '''
    
    
    
    neighbors = [12,14,16]
    estimators = [50,100,150]
    df = pd.DataFrame(columns=['Neighbors','Estimators','Confusion Matrix'])
    rows = []
    for n in neighbors:
        estimator = KNeighborsClassifier(n_neighbors=n,n_jobs=-1)
        for n_est in estimators:

            bag = BaggingClassifier(estimator, n_estimators=n_est,  random_state=0)
            bag.fit(X_train,y_train)
            predicted_labels = bag.predict(X_test)
            tn, fp, fn, tp = confusion_matrix(y_test, predicted_labels, labels=[0,1]).ravel()
            convert_matrix = [tn,fp,fn,tp]
            rows.append([n,n_est,convert_matrix])

    for i in range(len(rows)):
        df = df.append({'Neighbors':rows[i][0],'Estimators':rows[i][1],'Confusion Matrix':rows[i][2]}, ignore_index=True)

    return df

def bag_SVM(X_train,X_test,y_train,y_test):
        '''
    applying bagging to logistic regression with elasticnet
    Args:
        X_train, X_test, y_train, y_test

    Returns:
        DataFrame: Preprocessed DataFrame. where Kernel, C, Gamma and degree are hyperparameters of SVM, estimator is hyperparameter for bagging, confusion matrix is the confusion matrix for each combination of those hyperparameters
    '''
    
    
    df = pd.DataFrame(columns=['Kernel','C','Gamma','Degree','Estimators','Confusion Matrix'])
    rows = []
    estimators = [50]#,100,150]

    Cs = [1e-1,1, 1e1, 1e2, 1e3]#,
    gammas = [1,1e1]
    degrees = [2,3]#,3
    for c in Cs:
        estimator = LinearSVC(C=c, random_state=0, max_iter=10000)
        for n_est in estimators:

            bag = BaggingClassifier(estimator, n_estimators=n_est, random_state=0)
            bag.fit(X_train, y_train)
            predicted_labels = bag.predict(X_test)
            tn, fp, fn, tp = confusion_matrix(y_test, predicted_labels, labels=[0,1]).ravel()
            convert_matrix = [tn,fp,fn,tp]
            rows.append(['linear', c, '', '', n_est,convert_matrix])

        for gamma in gammas:
            estimator = SVC(kernel = 'rbf', C=c, gamma=gamma, random_state=0, max_iter=10000)
            for n_est in estimators:

                bag = BaggingClassifier(estimator, n_estimators=n_est, random_state=0)
                bag.fit(X_train, y_train)
                predicted_labels = bag.predict(X_test)
                tn, fp, fn, tp = confusion_matrix(y_test, predicted_labels, labels=[0,1]).ravel()
                convert_matrix = [tn,fp,fn,tp]
                rows.append(['rbf', c, gamma, '', n_est,convert_matrix])

            for degree in degrees:
                estimator = SVC(kernel='poly', C=c, gamma=gamma, degree=degree, random_state=0, max_iter=10000)
                for n_est in estimators:

                        bag = BaggingClassifier(estimator, n_estimators=n_est,  random_state=0)
                        bag.fit(X_train, y_train)
                        predicted_labels = bag.predict(X_test)
                        tn, fp, fn, tp = confusion_matrix(y_test, predicted_labels, labels=[0,1]).ravel()
                        convert_matrix = [tn,fp,fn,tp]


                        rows.append(['poly', c, gamma, degree, n_est,convert_matrix])

    for i in range(len(rows)):
        df = df.append({'Kernel':rows[i][0],'C':rows[i][1],'Gamma':rows[i][2], 'Degree':rows[i][3],'Estimators': rows[i][4],
        'Confusion Matrix':rows[i][5]}, ignore_index=True)
    return df


def bag_rdforest(X_train,X_test,y_train,y_test):
    
        '''
    applying bagging to logistic regression with elasticnet
    Args:
        X_train, X_test, y_train, y_test

    Returns:
        DataFrame: Preprocessed DataFrame. where N_Estimators and Max_Depth are hyperparameters of Random Forest, estimator is hyperparameter for bagging, confusion matrix is the confusion matrix for each combination of those hyperparameters
    '''
    
    estimators = [50,100,150]

    df = pd.DataFrame(columns=['N_Estimators','Max_Depth','Estimators','Confusion Matrix'])
    rows = []

    n_estimators = [300, 500]
    max_depths = [5,10,15]

    for estimator in n_estimators:
        for max_d in max_depths:
            classifier = RandomForestClassifier(n_estimators=estimator, max_depth=max_d,  random_state=0, n_jobs=-1)
            for n_est in estimators:
                bag = BaggingClassifier(classifier, n_estimators=n_est,  random_state=0)
                bag.fit(X_train,y_train)
                predicted_labels = bag.predict(X_test)
                tn, fp, fn, tp = confusion_matrix(y_test, predicted_labels, labels=[0,1]).ravel()
                convert_matrix = [tn,fp,fn,tp]



                rows.append([estimator,max_d,n_est,  convert_matrix])


    for i in range(len(rows)):
        df = df.append({'N_Estimators':rows[i][0],'Max_Depth':rows[i][1],'Estimators':rows[i][2],'Confusion Matrix':rows[i][3]}, ignore_index=True)
    return df


def bag_xgboost(X_train,X_test,y_train,y_test):
    '''
    applying bagging to logistic regression with elasticnet
    Args:
        X_train, X_test, y_train, y_test

    Returns:
        DataFrame: Preprocessed DataFrame. where Max_depth and N_estimators are hyperparameters of XGBoost, estimators  and samples are hyperparameter for bagging, confusion matrix is the confusion matrix for each combination of those hyperparameters
    '''
    
    
    
    df = pd.DataFrame(columns=['Max_depth','N_estimators','Estimators','Samples','Confusion Matrix'])
    rows = []
    therate = 0.05
    max_depth = [1,2,3,4]#np.linspace(2, 10, 5, dtype=int)
    n_estimators= [50,100,150]#np.linspace(50, 450, 4, dtype=int)
    estimators = [50,100,150]
    samples = [0.5,0.75,1]
    for depth in max_depth:
        for estimators in n_estimators:
            estimator = XGBClassifier(booster='gbtree',max_depth=depth,learning_rate=therate,n_estimators = estimators)
            for n_est in estimators:
                for sample in samples:
                    bag = BaggingClassifier(estimator, n_estimators=n_est, max_samples=sample, random_state=0)
                    bag.fit(X_train,y_train)
                    predicted_labels = bag.predict(X_test)
                    tn, fp, fn, tp = confusion_matrix(y_test, predicted_labels, labels=[0,1]).ravel()
                    convert_matrix = [tn,fp,fn,tp]
                    rows.append([depth,estimators,n_est, sample, convert_matrix])

    for i in range(len(rows)):
        df = df.append({'Max_depth':rows[i][0],'N_estimators':rows[i][1],'Estimators':rows[i][2],'Samples':rows[i][3],
                        'Confusion Matrix':rows[i][4]}, ignore_index=True)

    return df

def bag_naive_bayes(X_train,X_test,y_train,y_test):
        '''
    applying bagging to logistic regression with elasticnet
    Args:
        X_train, X_test, y_train, y_test

    Returns:
        DataFrame: Preprocessed DataFrame. where Gauss indicate whether it is Gauss(1) or Bernoulli(0), estimator is hyperparameter for bagging, confusion matrix is the confusion matrix for each combination of those hyperparameters
    '''
    
    estimators = [50,100,150]

    df = pd.DataFrame(columns=['Gauss', 'Estimators','Confusion Matrix'])
    rows = []

    estimator = GaussianNB()
    for n_est in estimators:

        bag = BaggingClassifier(estimator, n_estimators=n_est, random_state=0)
        bag.fit(X_train,y_train)
        predicted_labels = bag.predict(X_test)
        tn, fp, fn, tp = confusion_matrix(y_test, predicted_labels, labels=[0,1]).ravel()
        convert_matrix = [tn,fp,fn,tp]
        rows.append([1,n_est, convert_matrix])
    estimator = BernoulliNB()
    for n_est in estimators:

        bag = BaggingClassifier(estimator, n_estimators=n_est,random_state=0)
        bag.fit(X_train,y_train)
        predicted_labels = bag.predict(X_test)
        tn, fp, fn, tp = confusion_matrix(y_test, predicted_labels, labels=[0,1]).ravel()
        convert_matrix = [tn,fp,fn,tp]
        rows.append([0,n_est,  convert_matrix])
    for i in range(len(rows)):
        df = df.append({'Gauss':rows[i][0],'Estimators':rows[i][1],'Confusion Matrix':rows[i][2]}, ignore_index=True)
    return df


def classify_bag(estimator, X_train, X_test, y_train, y_test, n_est=None, rate=None):
        '''
    choosing which classifier to do bagging
    Args:
        estimator is the estimator to do bagging on

    Returns:
        results of bagging by that chosen estimator
    '''
    
    
    
    if estimator == 'svm':
        return bag_SVM(X_train, X_test, y_train, y_test)
    elif estimator == 'naive_bayes':
        return bag_naive_bayes(X_train, X_test, y_train, y_test)
    elif estimator == 'rdforest':
        return bag_rdforest(X_train, X_test, y_train, y_test)
    elif estimator == 'knn':
        return bag_KNN(X_train, X_test, y_train, y_test)

    elif estimator =='xgboost':
        return bag_xgboost(X_train, X_test, y_train, y_test)
    elif estimator == 'elasticnet':
        return  bag_elasticnet(X_train, X_test, y_train, y_test)