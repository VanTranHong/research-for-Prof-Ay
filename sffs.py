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


def KNN_subset(X_train,X_test,y_train,y_test, features):
    index =0
    neighbors = [5,10,12,14,16,20]
    df = pd.DataFrame(columns=['Neighbors','Confusion Matrix'])
    rows = []

    for n in neighbors:
        feature = features[index]
        knn = KNeighborsClassifier(n_neighbors=n,n_jobs=-1)
        X_train_knn, X_test_knn = X_train[:,feature], X_test[:,feature]
        knn.fit(X_train_knn,y_train)
        predicted_labels = knn.predict(X_test_knn)
        tn, fp, fn, tp = confusion_matrix(y_test, predicted_labels, labels=[0,1]).ravel()
        convert_matrix = [tn,fp,fn,tp]
        rows.append([n, convert_matrix])
        index +=1

    for i in range(len(rows)):
        df = df.append({'Neighbors':rows[i][0],'Confusion Matrix':rows[i][1]}, ignore_index=True)

    return df

def elastic_subset(X_train,X_test,y_train,y_test, features):
    
    index =0
   
    
    df = pd.DataFrame(columns=['Alpha','L1 ratio','Confusion Matrix'])
    rows = []
    alphas= [0.0001, 0.001]#, 0.01, 0.1, 1, 10, 100]
    l1s = [0.0,0.1,0.2]#,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    for al in alphas:
        for l1 in l1s:
            feature = features[index]
            regr = SGDClassifier(loss = 'log',alpha= al,penalty = 'elasticnet',l1_ratio=l1,random_state=0)
            X_train_knn, X_test_knn = X_train[:,feature], X_test[:,feature]
            regr.fit(X_train_knn,y_train)
            predicted_labels = regr.predict(X_test_knn)
            tn, fp, fn, tp = confusion_matrix(y_test, predicted_labels, labels=[0,1]).ravel()
            convert_matrix = [tn,fp,fn,tp]
            rows.append([al,l1, convert_matrix])
            index +=1
    for i in range(len(rows)):
        df = df.append({'Alpha':rows[i][0],'L1 ratio':rows[i][1],'Confusion Matrix':rows[i][2]}, ignore_index=True)

    return df



def SVM_subset(X_train, X_test, y_train, y_test,features):
    index =0
    df = pd.DataFrame(columns=['Kernel','C','Gamma','Degree','Confusion Matrix'])
    rows = []

    Cs = [1e-1, 1, 1e1, 1e2, 1e3]
    gammas = [1,1e1]
    degrees = [2,3]

    for c in Cs:
        feature = features[index]
        linear = LinearSVC(C=c, random_state=0, max_iter=10000)

        X_train_linear, X_test_linear = X_train[:,feature], X_test[:,feature]

        linear.fit(X_train_linear, y_train)
        predicted_labels = linear.predict(X_test_linear)
        tn, fp, fn, tp = confusion_matrix(y_test, predicted_labels, labels=[0,1]).ravel()
        convert_matrix = [tn,fp,fn,tp]
        rows.append(['linear', c, '', '',convert_matrix])
        index+=1

        for gamma in gammas:
            feature = features[index]
            rbf = SVC(kernel = 'rbf', C=c, gamma=gamma, random_state=0, max_iter=10000)

            X_train_rbf, X_test_rbf = X_train[:,feature], X_test[:,feature]

            rbf.fit(X_train_rbf, y_train)
            predicted_labels = rbf.predict(X_test_rbf)
            tn, fp, fn, tp = confusion_matrix(y_test, predicted_labels, labels=[0,1]).ravel()
            convert_matrix = [tn,fp,fn,tp]
            rows.append(['rbf', c, gamma, '', convert_matrix])
            index+=1

            for degree in degrees:
                feature = features[index]
                poly = SVC(kernel='poly', C=c, gamma=gamma, degree=degree, random_state=0, max_iter=10000)
                X_train_poly, X_test_poly = X_train[:,feature], X_test[:,feature]

                poly.fit(X_train_poly,y_train)
                predicted_labels = poly.predict(X_test_poly)
                tn, fp, fn, tp = confusion_matrix(y_test, predicted_labels, labels=[0,1]).ravel()
                convert_matrix = [tn,fp,fn,tp]
                rows.append(['poly', c, gamma, degree, convert_matrix])
                index+=1

    for i in range(len(rows)):
        df = df.append({'Kernel':rows[i][0],'C':rows[i][1],'Gamma':rows[i][2], 'Degree':rows[i][3],
        'Confusion Matrix':rows[i][4]}, ignore_index=True)

    return df


def rdforest_subset(X_train,X_test,y_train,y_test,features):
    index =0
    df = pd.DataFrame(columns=['N_Estimators','Max_Depth','Confusion Matrix'])
    rows = []

    estimators = [ 300, 500]
    max_depths = [5,10,15]

    for estimator in estimators:
        for max_d in max_depths:
            feature = features[index]
            rdf = RandomForestClassifier(n_estimators=estimator, max_depth=max_d,  random_state=0, n_jobs=-1)

            X_train_rdf, X_test_rdf = X_train[:,feature], X_test[:,feature]
            rdf.fit(X_train_rdf, y_train)
            predicted_labels = rdf.predict(X_test_rdf)
            tn, fp, fn, tp = confusion_matrix(y_test, predicted_labels, labels=[0,1]).ravel()
            convert_matrix = [tn,fp,fn,tp]
            rows.append([estimator, max_d,  convert_matrix])
            index+=1

    for i in range(len(rows)):
        df = df.append({'N_Estimators':rows[i][0],'Max_Depth':rows[i][1],
        'Confusion Matrix':rows[i][2]}, ignore_index=True)

    return df



def xgboost_subset(X_train,X_test,y_train,y_test,features):
    index =0
    df = pd.DataFrame(columns=['Max_depth','N_estimators','Confusion Matrix'])
    # ,'N_estimators','ColSample','Subsample',
    # 'Min_Child_Weight','Scale_Pos_Weight',
    rows = []

    rate = 0.05
    #xg_alpha = 0
    #xg_lambda = 1
    max_depth = [1,2,3,4]#np.linspace(2, 10, 5, dtype=int)
    n_estimators= [50,100,150]#np.linspace(50, 450, 4, dtype=int)
    # colsample_bytree= np.linspace(0.5, 1, 4)
    # subsample = np.linspace(0.5,1,4)
    # min_child_weight = np.linspace(1, 19, 4, dtype=int)
    # scale_pos_weight=np.linspace(3, 10,4, dtype=int)

    for depth in max_depth:
        for estimators in n_estimators:
            feature = features[index]
        #     for col in colsample_bytree:
        #         for sub in subsample:
        #             for child_weight in min_child_weight:
        #                 for pos_weight in scale_pos_weight:
        # min_child_weight=child_weight,colsample_bytree=col,
        # scale_pos_weight=pos_weight,n_estimators=estimators, subsample=sub
            xgb = XGBClassifier(booster='gbtree',max_depth=depth,learning_rate=rate,n_estimators = estimators)
            X_train_xgb, X_test_xgb = X_train[:,feature], X_test[:,feature]
            xgb.fit(X_train_xgb, y_train)
            predicted_labels = xgb.predict(X_test_xgb)
            tn, fp, fn, tp = confusion_matrix(y_test, predicted_labels, labels=[0,1]).ravel()
            convert_matrix = [tn,fp,fn,tp]
            # ,estimators,col,sub,child_weight,pos_weight
            rows.append([depth,estimators,convert_matrix])
            index+=1

    for i in range(len(rows)):
        df = df.append({'Max_depth':rows[i][0],'N_estimators':rows[i][1],
                        'Confusion Matrix':rows[i][2]}, ignore_index=True)

    return df


def naive_bayes_subset(X_train,X_test,y_train,y_test, features):
    index =0
    df = pd.DataFrame(columns=['Gauss', 'Confusion Matrix'])
    rows = []

    gnb = GaussianNB()
    feature = features[index]
    index+=1
    X_train_gnb, X_test_gnb = X_train[:,feature], X_test[:,feature]
    gnb.fit(X_train_gnb, y_train)
    predicted_labels = gnb.predict(X_test_gnb)
    tn_g, fp_g, fn_g, tp_g = confusion_matrix(y_test, predicted_labels, labels=[0,1]).ravel()
    convert_matrix_g = [tn_g,fp_g,fn_g,tp_g]

    df = df.append({'Gauss':1, 'Confusion Matrix':convert_matrix_g}, ignore_index=True)

    bnb = BernoulliNB()
    feature = features[index]


    X_train_bnb, X_test_bnb = X_train[:,feature], X_test[:,feature]
    bnb.fit(X_train_bnb, y_train)
    predicted_labels = bnb.predict(X_test_bnb)
    tn, fp, fn, tp = confusion_matrix(y_test, predicted_labels, labels=[0,1]).ravel()
    convert_matrix_b = [tn,fp,fn,tp]

    df = df.append({'Gauss':0,'Confusion Matrix':convert_matrix_b}, ignore_index=True)


    return df

def classify_sfs(estimator, X_train, X_test, y_train, y_test, features):
    if estimator == 'svm':
        return SVM_subset(X_train, X_test, y_train, y_test, features)
    elif estimator == 'rdforest':
        return rdforest_subset(X_train, X_test, y_train, y_test, features)
    elif estimator == 'knn':
        return KNN_subset(X_train, X_test, y_train, y_test, features)
    elif estimator == 'naive_bayes':
        return naive_bayes_subset(X_train, X_test, y_train, y_test, features)
    elif estimator =='xgboost':
        return xgboost_subset(X_train, X_test, y_train, y_test, features)
    elif estimator == 'elasticnet':
        return elastic_subset(X_train, X_test, y_train, y_test, features)