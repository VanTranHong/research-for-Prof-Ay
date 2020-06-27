import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier

def KNN(X_train,X_test,y_train,y_test):
    neighbors = [1,3,5,7,9]
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

    Cs = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]
    gammas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1]
    degrees = [2,3,4,5]

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

def SVM_bag(X_train,X_test,y_train,y_test, n_est):

    df = pd.DataFrame(columns=['Kernel','C','Gamma','Degree','Bag_Estimators','Confusion Matrix'])
    rows = []

    Cs = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]
    gammas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1]
    degrees = [2,3,4,5]

    for c in Cs:
        linear = LinearSVC(C=c, random_state=0, max_iter=10000)
        abc = BaggingClassifier(n_estimators = n_est, base_estimator=linear, random_state=0)
        predicted_labels = abc.predict(X_test)
        tn, fp, fn, tp = confusion_matrix(y_test, predicted_labels, labels=[0,1]).ravel()
        convert_matrix = [tn,fp,fn,tp]
        rows.append(['linear', c, '', '', n_est, convert_matrix])

        for gamma in gammas:
            rbf = SVC(kernel = 'rbf', C=c, gamma=gamma, random_state=0, max_iter=10000)
            abc = BaggingClassifier(n_estimators = n_est, base_estimator=rbf, random_state=0)
            predicted_labels = abc.predict(X_test)
            tn, fp, fn, tp = confusion_matrix(y_test, predicted_labels, labels=[0,1]).ravel()
            convert_matrix = [tn,fp,fn,tp]
            rows.append(['rbf', c, gamma, '', n_est, convert_matrix])

            for degree in degrees:
                poly = SVC(kernel='poly', C=c, gamma=gamma, degree=degree, random_state=0, max_iter=10000)
                abc = BaggingClassifier(n_estimators = n_est, base_estimator=poly, random_state=0)
                abc.fit(X_train,y_train)
                predicted_labels = abc.predict(X_test)
                tn, fp, fn, tp = confusion_matrix(y_test, predicted_labels, labels=[0,1]).ravel()
                convert_matrix = [tn,fp,fn,tp]
                rows.append(['poly', c, gamma, degree, n_est, convert_matrix])

    for i in range(len(rows)):
        df = df.append({'Kernel':rows[i][0],'C':rows[i][1],'Gamma':rows[i][2], 'Degree':rows[i][3],
        'Bag_Estimators':rows[i][4],'Confusion Matrix':rows[i][5]}, ignore_index=True)

    return df

def rdforest(X_train,X_test,y_train,y_test):
    df = pd.DataFrame(columns=['N_Estimators','Max_Features','Confusion Matrix'])
    rows = []
    
    estimators = [100, 200,300, 400, 500]
    max_features = [5,10]

    for estimator in estimators:
        for max_feature in max_features:
            rdf = RandomForestClassifier(n_estimators=estimator, max_features=max_feature,  random_state=0)
            rdf.fit(X_train, y_train)
            predicted_labels = rdf.predict(X_test)
            tn, fp, fn, tp = confusion_matrix(y_test, predicted_labels, labels=[0,1]).ravel()
            convert_matrix = [tn,fp,fn,tp]
            rows.append([estimator, max_feature, convert_matrix])
    
    for i in range(len(rows)):
        df = df.append({'N_Estimators':rows[i][0],'Max_Features':rows[i][1],'Confusion Matrix':rows[i][2]}, ignore_index=True)

    return df

def rdforest_boost(X_train,X_test,y_train,y_test,n_est, rate):
    df = pd.DataFrame(columns=['N_Estimators','Max_Features','Boost_Estimators','Rate','Confusion Matrix'])
    rows = []
    
    estimators = [100, 200]
    max_features = [5,10]

    for estimator in estimators:
        for max_feature in max_features:
            rdf = RandomForestClassifier(n_estimators=estimator, max_features=max_feature,  random_state=0)
            abc = AdaBoostClassifier(n_estimators = n_est, base_estimator=rdf, learning_rate =rate)
            abc.fit(X_train, y_train)
            predicted_labels = abc.predict(X_test)
            tn, fp, fn, tp = confusion_matrix(y_test, predicted_labels, labels=[0,1]).ravel()
            convert_matrix = [tn,fp,fn,tp]
            rows.append([estimator, max_feature, n_est, rate, convert_matrix])
    
    for i in range(len(rows)):
        df = df.append({'N_Estimators':rows[i][0],'Max_Features':rows[i][1],
                        'Boost_Estimators':rows[i][2], 'Rate':rows[i][3],
                        'Confusion Matrix':rows[i][4]}, ignore_index=True)

    return df

def rdforest_bag(X_train,X_test,y_train,y_test,n_est):
    df = pd.DataFrame(columns=['N_Estimators','Max_Features','Bag_Estimators','Confusion Matrix'])
    rows = []
    
    estimators = [100, 200]
    max_features = [5]

    for estimator in estimators:
        for max_feature in max_features:
            rdf = RandomForestClassifier(n_estimators=estimator, max_features=max_feature,  random_state=0)
            abc = BaggingClassifier(n_estimators = n_est, base_estimator=rdf, random_state=0)
            abc.fit(X_train, y_train)
            predicted_labels = abc.predict(X_test)
            tn, fp, fn, tp = confusion_matrix(y_test, predicted_labels, labels=[0,1]).ravel()
            convert_matrix = [tn,fp,fn,tp]
            rows.append([estimator, max_feature, n_est, convert_matrix])
    
    for i in range(len(rows)):
        df = df.append({'N_Estimators':rows[i][0],'Max_Features':rows[i][1],
                        'Bag_Estimators':rows[i][2], 'Confusion Matrix':rows[i][3]}, ignore_index=True)

    return df

def xgboost(X_train,X_test,y_train,y_test):
    df = pd.DataFrame(columns=['Max_depth','N_estimators','ColSample','Subsample',
    'Min_Child_Weight','Scale_Pos_Weight','Confusion Matrix'])
    rows = []

    rate = 0.05
    #xg_alpha = 0
    #xg_lambda = 1
    max_depth = np.linspace(2, 10, 4, dtype=int)
    n_estimators= np.linspace(50, 450, 4, dtype=int)
    colsample_bytree= np.linspace(0.5, 1, 4)
    subsample = np.linspace(0.5,1,4)    
    min_child_weight = np.linspace(1, 19, 5, dtype=int)
    scale_pos_weight=np.linspace(3, 10,3, dtype=int)
    
    for depth in max_depth:
        for estimators in n_estimators:
            for col in colsample_bytree:
                for sub in subsample:
                    for child_weight in min_child_weight:
                        for pos_weight in scale_pos_weight:
                            xgb = XGBClassifier(booster='gbtree',learning_rate=rate,max_depth=depth,
                                                min_child_weight=child_weight,colsample_bytree=col,
                                                scale_pos_weight=pos_weight,n_estimators=estimators, subsample=sub)
                            xgb.fit(X_train, y_train)
                            predicted_labels = xgb.predict(X_test)
                            tn, fp, fn, tp = confusion_matrix(y_test, predicted_labels, labels=[0,1]).ravel()
                            convert_matrix = [tn,fp,fn,tp]
                            rows.append([depth,estimators,col,sub,child_weight,pos_weight,convert_matrix])
    
    for i in range(len(rows)):
        df = df.append({'Max_depth':rows[i][0],'N_estimators':rows[i][1],'ColSample':rows[i][2],
                        'Subsample':rows[i][3],'Min_Child_Weight':rows[i][4],'Scale_Pos_Weight':rows[i][5],
                        'Confusion Matrix':rows[i][6]}, ignore_index=True)

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
    elif estimator == 'rdforest_boost':
        return rdforest_boost(X_train, X_test, y_train, y_test, n_est, rate)
    elif estimator == 'rdforest_bag':
        return rdforest_bag(X_train, X_test, y_train, y_test, n_est)
    elif estimator =='xgboost':
        return xgboost(X_train, X_test, y_train, y_test)

#def stats & univariate stats
