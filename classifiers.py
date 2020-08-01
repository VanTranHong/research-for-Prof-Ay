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


def elasticnet(X_train,X_test,y_train,y_test):
    df = pd.DataFrame(columns=['Alpha','Confusion Matrix'])
    rows = []
    alphas= [0.0001, 0.001, 0.01]
    

    for al in alphas:
        

        regr = SGDClassifier(loss = 'log',alpha= al,penalty = 'l1',random_state=0)

        model = regr.fit(X_train, y_train)
        predicted_labels = model.predict(X_test)
        tn, fp, fn, tp = confusion_matrix(y_test, predicted_labels, labels=[0,1]).ravel()
        convert_matrix = [tn,fp,fn,tp]
        rows.append([al,convert_matrix])
    for i in range(len(rows)):
        df = df.append({'Alpha':rows[i][0],'Confusion Matrix':rows[i][1]}, ignore_index=True)

    return df

def KNN(X_train,X_test,y_train,y_test):
    neighbors = [5,10,12,14,16,20]
    df = pd.DataFrame(columns=['Neighbors','Confusion Matrix'])
    rows = []

    for n in neighbors:
        knn = KNeighborsClassifier(n_neighbors=n,n_jobs=-1)
        knn.fit(X_train,y_train)
        predicted_labels = knn.predict(X_test)
        tn, fp, fn, tp = confusion_matrix(y_test, predicted_labels, labels=[0,1]).ravel()
        convert_matrix = [tn,fp,fn,tp]
        rows.append([n, convert_matrix])

    for i in range(len(rows)):
        df = df.append({'Neighbors':rows[i][0],'Confusion Matrix':rows[i][1]}, ignore_index=True)

    return df


def SVM(X_train,X_test,y_train,y_test):

    df = pd.DataFrame(columns=['Kernel','C','Gamma','Degree','Confusion Matrix'])
    rows = []

    Cs = [1e-1, 1, 1e1, 1e2, 1e3]
    gammas = [1,1e1]
    degrees = [2,3]

    for c in Cs:
        linear = LinearSVC(C=c, random_state=0, max_iter=10000)
        linear.fit(X_train, y_train)
        predicted_labels = linear.predict(X_test)
        tn, fp, fn, tp = confusion_matrix(y_test, predicted_labels, labels=[0,1]).ravel()
        convert_matrix = [tn,fp,fn,tp]
        rows.append(['linear', c, '', '', convert_matrix])

        for gamma in gammas:
            rbf = SVC(kernel = 'rbf', C=c, gamma=gamma, random_state=0, max_iter=10000)
            rbf.fit(X_train, y_train)
            predicted_labels = rbf.predict(X_test)
            tn, fp, fn, tp = confusion_matrix(y_test, predicted_labels, labels=[0,1]).ravel()
            convert_matrix = [tn,fp,fn,tp]
            rows.append(['rbf', c, gamma, '', convert_matrix])

            for degree in degrees:

                poly = SVC(kernel='poly', C=c, gamma=gamma, degree=degree, random_state=0, max_iter=10000)
                poly.fit(X_train,y_train)
                predicted_labels = poly.predict(X_test)
                tn, fp, fn, tp = confusion_matrix(y_test, predicted_labels, labels=[0,1]).ravel()
                convert_matrix = [tn,fp,fn,tp]
                rows.append(['poly', c, gamma, degree, convert_matrix])

    for i in range(len(rows)):
        df = df.append({'Kernel':rows[i][0],'C':rows[i][1],'Gamma':rows[i][2], 'Degree':rows[i][3],
        'Confusion Matrix':rows[i][4]}, ignore_index=True)

    return df



def rdforest(X_train,X_test,y_train,y_test):
    df = pd.DataFrame(columns=['N_Estimators','Max_Depth','Confusion Matrix'])
    rows = []

    estimators = [300, 500]
    max_depths = [5,10,15]

    for estimator in estimators:
        for max_d in max_depths:
            rdf = RandomForestClassifier(n_estimators=estimator, max_depth=max_d,  random_state=0, n_jobs=-1)
            rdf.fit(X_train, y_train)
            predicted_labels = rdf.predict(X_test)
            tn, fp, fn, tp = confusion_matrix(y_test, predicted_labels, labels=[0,1]).ravel()
            convert_matrix = [tn,fp,fn,tp]
            rows.append([estimator, max_d, convert_matrix])

    for i in range(len(rows)):
        df = df.append({'N_Estimators':rows[i][0],'Max_Depth':rows[i][1],'Confusion Matrix':rows[i][2]}, ignore_index=True)

    return df


def xgboost(X_train,X_test,y_train,y_test):
    df = pd.DataFrame(columns=['Max_depth','N_estimators','Confusion Matrix'])
    # ,'N_estimators','ColSample','Subsample',
    # 'Min_Child_Weight','Scale_Pos_Weight',
    rows = []

    rate = 0.05

    max_depth = [1,2,3,4]#np.linspace(2, 10, 5, dtype=int)
    n_estimators= [50,100,150]#np.linspace(50, 450, 4, dtype=int)


    for depth in max_depth:
        for estimators in n_estimators:
            xgb = XGBClassifier(booster='gbtree',max_depth=depth,learning_rate=rate,n_estimators = estimators)
            xgb.fit(X_train, y_train)
            predicted_labels = xgb.predict(X_test)
            tn, fp, fn, tp = confusion_matrix(y_test, predicted_labels, labels=[0,1]).ravel()
            convert_matrix = [tn,fp,fn,tp]
            # ,estimators,col,sub,child_weight,pos_weight
            rows.append([depth,estimators,convert_matrix])

    for i in range(len(rows)):
        df = df.append({'Max_depth':rows[i][0],'N_estimators':rows[i][1],
                        'Confusion Matrix':rows[i][2]}, ignore_index=True)

    return df


def naive_bayes(X_train,X_test,y_train,y_test):
    df = pd.DataFrame(columns=['Gauss', 'Confusion Matrix'])
    rows = []

    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    predicted_labels = gnb.predict(X_test)
    tn_g, fp_g, fn_g, tp_g = confusion_matrix(y_test, predicted_labels, labels=[0,1]).ravel()
    convert_matrix_g = [tn_g,fp_g,fn_g,tp_g]

    df = df.append({'Gauss':1, 'Confusion Matrix':convert_matrix_g}, ignore_index=True)

    bnb = BernoulliNB()
    bnb.fit(X_train, y_train)
    predicted_labels = bnb.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, predicted_labels, labels=[0,1]).ravel()
    convert_matrix_b = [tn,fp,fn,tp]

    df = df.append({'Gauss':0, 'Confusion Matrix':convert_matrix_b}, ignore_index=True)


    return df



def classify(estimator, X_train, X_test, y_train, y_test, n_est=None, rate=None):
    if estimator == 'svm':
        return SVM(X_train, X_test, y_train, y_test)
    elif estimator == 'naive_bayes':
        return naive_bayes(X_train, X_test, y_train, y_test)
    elif estimator == 'rdforest':
        return rdforest(X_train, X_test, y_train, y_test)
    elif estimator == 'knn':
        return KNN(X_train, X_test, y_train, y_test)
    elif estimator == 'elasticnet':
        return elasticnet(X_train, X_test, y_train, y_test)

    elif estimator =='xgboost':
        return xgboost(X_train, X_test, y_train, y_test)



