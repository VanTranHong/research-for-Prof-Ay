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


def boost_elasticnet(X_train,X_test,y_train,y_test):
    
  
    # applying bagging to logistic regression with elasticnet
    # Args:
    #     X_train, X_test, y_train, y_test

    # Returns:
    #     DataFrame: Preprocessed DataFrame. where Alpha and L1 ratio are hyperparameters of elastic net, estimator is hyperparameter for bagging, confusion matrix is the confusion matrix for each combination of those hyperparameters
    
    
    df = pd.DataFrame(columns=['Alpha','Estimators','Learning Rate','Confusion Matrix'])
    rows = []
    alphas= [0.0001, 0.001, 0.01]#,0.1,1]
    estimators = [50,100,150]
    rates = [0.5,0.75,1]
    
    

    for al in alphas:
        estimator = SGDClassifier(loss = 'log',alpha= al,penalty = 'l1',random_state=0)
        for n_est in estimators:
            for rate in rates:
                ada = AdaBoostClassifier(estimator,n_estimators=n_est,algorithm='SAMME', learning_rate=rate, random_state=0) #
                ada.fit(X_train,y_train)
                predicted_labels = ada.predict(X_test)
                tn, fp, fn, tp = confusion_matrix(y_test, predicted_labels, labels=[0,1]).ravel()
                convert_matrix = [tn,fp,fn,tp]
                rows.append([al,n_est, rate, convert_matrix])

            
            
    for i in range(len(rows)):
        df = df.append({'Alpha':rows[i][0],'Estimators':rows[i][1],'Learning Rate':rows[i][2],'Confusion Matrix':rows[i][3]}, ignore_index=True)

    return df

def boost_KNN(X_train,X_test,y_train,y_test):
      
    # '''
    # applying boosting to KNN classifier
    # Args:
    #     X_train, X_test, y_train, y_test

    # Returns:
    #     DataFrame: Preprocessed DataFrame. where neighbors is hyperparameters of KNN, learning rate, estimator is hyperparameter for boosting, confusion matrix is the confusion matrix for each combination of those hyperparameters
    # '''
    
    
    
    
    estimators = [50,100,150]
    rates = [0.5,0.75,1]
    neighbors = [5,10,12,14,16,20]
    df = pd.DataFrame(columns=['Neighbors','Estimators','Learning Rate','Confusion Matrix'])

    rows = []
    for n in neighbors:
        estimator = KNeighborsClassifier(n_neighbors=n,n_jobs=-1)
        for n_est in estimators:
            for rate in rates:
                ada = AdaBoostClassifier(estimator, algorithm='SAMME',n_estimators=n_est, learning_rate=rate, random_state=0)
                ada.fit(X_train,y_train)
                predicted_labels = ada.predict(X_test)
                tn, fp, fn, tp = confusion_matrix(y_test, predicted_labels, labels=[0,1]).ravel()
                convert_matrix = [tn,fp,fn,tp]
                rows.append([n,n_est, rate, convert_matrix])

    for i in range(len(rows)):
        df = df.append({'Neighbors':rows[i][0],'Estimators':rows[i][1],'Learning Rate':rows[i][2],'Confusion Matrix':rows[i][3]}, ignore_index=True)
    return df

def boost_SVM(X_train,X_test,y_train,y_test):
    # '''
    # applying boosting to SVM classifier
    # Args:
    #     X_train, X_test, y_train, y_test

    # Returns:
    #     DataFrame: Preprocessed DataFrame. where Kernal, C, gamma, degrees are hyperparameters of SVM, learning rate, estimator is hyperparameter for boosting, confusion matrix is the confusion matrix for each combination of those hyperparameters
    # '''
    
    
    
    
    df = pd.DataFrame(columns=['Kernel','C','Gamma','Degree','Estimators','Learning Rate','Confusion Matrix'])
    rows = []
    estimators = [50,100,150]
    rates = [0.5,0.75,1]
    Cs = [1e-1, 1, 1e1, 1e2, 1e3]
    gammas = [1,1e1]
    degrees = [2,3]
    for c in Cs:
        estimator = LinearSVC(C=c, random_state=0, max_iter=10000)
        for n_est in estimators:
            for rate in rates:
                ada = AdaBoostClassifier(estimator, algorithm='SAMME',n_estimators=n_est, learning_rate=rate, random_state=0)
                ada.fit(X_train, y_train)
                predicted_labels = ada.predict(X_test)
                tn, fp, fn, tp = confusion_matrix(y_test, predicted_labels, labels=[0,1]).ravel()
                convert_matrix = [tn,fp,fn,tp]
                rows.append(['linear', c, '', '', n_est,rate,convert_matrix])

        for gamma in gammas:
            estimator = SVC(kernel = 'rbf', C=c, gamma=gamma, random_state=0, max_iter=10000)
            for n_est in estimators:
                for rate in rates:
                    ada = AdaBoostClassifier(estimator,algorithm='SAMME', n_estimators=n_est, learning_rate=rate, random_state=0)

                    ada.fit(X_train, y_train)
                    predicted_labels = ada.predict(X_test)
                    tn, fp, fn, tp = confusion_matrix(y_test, predicted_labels, labels=[0,1]).ravel()
                    convert_matrix = [tn,fp,fn,tp]
                    rows.append(['rbf', c, gamma, '', n_est,rate,convert_matrix])

            for degree in degrees:


                estimator = SVC(kernel='poly', C=c,  gamma=gamma, degree=degree, random_state=0, max_iter=10000)
                for n_est in estimators:
                    for rate in rates:
                        ada = AdaBoostClassifier(estimator, algorithm='SAMME', n_estimators=n_est, learning_rate=rate, random_state=0)

                        ada.fit(X_train, y_train)
                        predicted_labels = ada.predict(X_test)
                        tn, fp, fn, tp = confusion_matrix(y_test, predicted_labels, labels=[0,1]).ravel()
                        convert_matrix = [tn,fp,fn,tp]

                        rows.append(['poly', c, gamma, degree, n_est,rate,convert_matrix])

    for i in range(len(rows)):
        df = df.append({'Kernel':rows[i][0],'C':rows[i][1],'Gamma':rows[i][2], 'Degree':rows[i][3],'Estimators': rows[i][4],'Learning Rate': rows[i][5],
        'Confusion Matrix':rows[i][6]}, ignore_index=True)
    return df


def boost_rdforest(X_train,X_test,y_train,y_test):
    #     '''
    # applying boosting to random forest classifier
    # Args:
    #     X_train, X_test, y_train, y_test

    # Returns:
    #     DataFrame: Preprocessed DataFrame. where n_estimators and max_depths are hyperparameters of random forest, rates, estimators are hyperparameter for boosting, confusion matrix is the confusion matrix for each combination of those hyperparameters
    # '''
    
    
    estimators = [50,100,150]
    rates = [0.5,0.75,1]
    df = pd.DataFrame(columns=['N_Estimators','Max_Depth','Estimators','Learning Rate','Confusion Matrix'])
    rows = []

    n_estimators = [300, 500]
    max_depths = [5,10,15]

    for estimator in n_estimators:
        for max_d in max_depths:
            classifier = RandomForestClassifier(n_estimators=estimator, max_depth=max_d,  random_state=0, n_jobs=-1)
            for n_est in estimators:
                for rate in rates:
                    ada = AdaBoostClassifier(classifier, n_estimators=n_est, learning_rate=rate, random_state=0)
                    ada.fit(X_train,y_train)
                    predicted_labels = ada.predict(X_test)
                    tn, fp, fn, tp = confusion_matrix(y_test, predicted_labels, labels=[0,1]).ravel()
                    convert_matrix = [tn,fp,fn,tp]
                    rows.append([estimator,max_d,n_est, rate, convert_matrix])


    for i in range(len(rows)):
        df = df.append({'N_Estimators':rows[i][0],'Max_Depth':rows[i][1],'Estimators':rows[i][2],'Learning Rate':rows[i][3],'Confusion Matrix':rows[i][4]}, ignore_index=True)
    return df



def boost_xgboost(X_train,X_test,y_train,y_test):
    # '''
    # applying boosting to XGBoost classifier
    # Args:
    #     X_train, X_test, y_train, y_test

    # Returns:
    #     DataFrame: Preprocessed DataFrame. where max_depth and n_estimators are hyperparameters of XGBoost, rates, estimators are hyperparameter for boosting, confusion matrix is the confusion matrix for each combination of those hyperparameters
    # '''
    
    
    
    
    df = pd.DataFrame(columns=['Max_depth','N_estimators','Estimators','Learning Rate','Confusion Matrix'])
    rows = []
    therate = 0.05
    max_depth = [1,2,3,4]#np.linspace(2, 10, 5, dtype=int)
    n_estimators= [50,100,150]#np.linspace(50, 450, 4, dtype=int)
    estimators = [50,100,150]
    rates = [0.5,0.75,1]
    for depth in max_depth:
        for estimators in n_estimators:
            estimator = XGBClassifier(booster='gbtree',max_depth=depth,learning_rate=therate,n_estimators = estimators)
            for n_est in estimators:
                for rate in rates:
                    ada = AdaBoostClassifier(estimator, n_estimators=n_est, learning_rate=rate, random_state=0)
                    ada.fit(X_train,y_train)
                    predicted_labels = ada.predict(X_test)
                    tn, fp, fn, tp = confusion_matrix(y_test, predicted_labels, labels=[0,1]).ravel()
                    convert_matrix = [tn,fp,fn,tp]
                    rows.append([depth,estimators,n_est, rate, convert_matrix])

    for i in range(len(rows)):
        df = df.append({'Max_depth':rows[i][0],'N_estimators':rows[i][1],'Estimators':rows[i][2],'Learning Rate':rows[i][3],
                        'Confusion Matrix':rows[i][4]}, ignore_index=True)

    return df


def boost_naive_bayes(X_train,X_test,y_train,y_test):
    #     '''
    # applying boosting to KNN classifier
    # Args:
    #     X_train, X_test, y_train, y_test

    # Returns:
    #     DataFrame: Preprocessed DataFrame. where Gauss indicate if it is Gauss(1) or Bernoulli(0), learning rates, estimators are hyperparameter for boosting, confusion matrix is the confusion matrix for each combination of those hyperparameters
    # '''
    
    
    
    estimators = [50,100,150]
    rates = [0.5,0.75,1]
    df = pd.DataFrame(columns=['Gauss', 'Estimators','Learning Rate','Confusion Matrix'])
    rows = []

    estimator = GaussianNB()
    for n_est in estimators:
        for rate in rates:
            ada = AdaBoostClassifier(estimator, n_estimators=n_est, learning_rate=rate, random_state=0)
            ada.fit(X_train,y_train)
            predicted_labels = ada.predict(X_test)
            tn, fp, fn, tp = confusion_matrix(y_test, predicted_labels, labels=[0,1]).ravel()
            convert_matrix = [tn,fp,fn,tp]
            rows.append([1,n_est, rate, convert_matrix])
    estimator = BernoulliNB()
    for n_est in estimators:
        for rate in rates:
            ada = AdaBoostClassifier(estimator, n_estimators=n_est, learning_rate=rate, random_state=0)
            ada.fit(X_train,y_train)
            predicted_labels = ada.predict(X_test)
            tn, fp, fn, tp = confusion_matrix(y_test, predicted_labels, labels=[0,1]).ravel()
            convert_matrix = [tn,fp,fn,tp]
            rows.append([0,n_est, rate, convert_matrix])
    for i in range(len(rows)):
        df = df.append({'Gauss':rows[i][0],'Estimators':rows[i][1],'Learning Rate':rows[i][2],'Confusion Matrix':rows[i][3]}, ignore_index=True)
    return df



def classify_boost(estimator, X_train, X_test, y_train, y_test, n_est=None, rate=None):
    
    if estimator == 'svm':
        return boost_SVM(X_train, X_test, y_train, y_test)
    elif estimator == 'naive_bayes':
        return boost_naive_bayes(X_train, X_test, y_train, y_test)
    elif estimator == 'rdforest':
        return boost_rdforest(X_train, X_test, y_train, y_test)
    elif estimator == 'knn':
        return boost_KNN(X_train, X_test, y_train, y_test)
    elif estimator =='xgboost':
        return boost_xgboost(X_train, X_test, y_train, y_test)
    elif estimator =='elasticnet':
        return boost_elasticnet(X_train, X_test, y_train, y_test)
    

