   
import pandas as pd
import numpy as np
import math
import csv
import sklearn
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Lasso, ElasticNet
from sklearn.datasets import make_regression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score, precision_recall_curve, auc
from statsmodels.stats.outliers_influence import summary_table
import statsmodels.api as sm
import pylab as pl
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import svm, datasets
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import scipy.stats as stats
from scipy.stats import chi2_contingency
from sklearn.feature_selection import SelectKBest 
from sklearn.feature_selection import chi2 
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import f_classif, chi2, mutual_info_classif
from statsmodels.stats.multicomp import pairwise_tukeyhsd   
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB

from featureselection import infogain, reliefF, sfs,run_feature_selection
from CFS import CFS
from data_preprocess import impute




############ boosting base classifier ###############

def boost (classifier):
    abc = AdaBoostClassifier(n_estimators = 50, base_estimator=classifier, learning_rate =0.1)
    return abc

########## bagging base classifier ################

def bag(classifier):
    abc = BaggingClassifier(base_estimator=classifier,n_estimators=50, random_state=0)
    return abc

   #______________________ALGORITHM______________________________ 
########## this is SVM##########
def impute(train_data,  test_data) :
    train_data_impute = []
   
    test_data_impute = []
   

    for i in range(5):
        
        imputer = IterativeImputer(sample_posterior=True, random_state=i, verbose=1)
        train_data_impute.append(imputer.fit_transform(train_data))
        test_data_impute.append(imputer.transform(test_data))
    train_data_impute = round(np.mean(train_data_impute, axis=0))
    test_data_impute = round(np.mean(test_data_impute, axis=0))
    
    return train_data_impute,test_data_impute

def KNN(data, column, features, method, file_to_write):
    skf = StratifiedKFold(n_splits=10)
    X = np.array(data.drop(columns = [column], axis = 1))
    y = np.array(data[column])
    top_features = {}
    dict = {}
    neighbors = [1,2,3,4,5,6,7,8,9,10]
    for neighbor in neighbors:
        dict.update(str{neighbor}:[0,0,0,0])
    columns = data.drop(columns = [column], axis = 1).columns 
    for column in columns:
        top_features.update({column:0})
    for train, test in skf.split(X,y):
        train_data, test_data = X[train], X[test]  
        train_data, test_data = impute(train_data, test_data)
        train_result, test_result = y[train], y[test] 
        train_data = pd.DataFrame(data = train_data, columns=columns)
        test_data = pd.DataFrame(data = test_data, columns=columns)
        features_selected = run_feature_selection(method,train_data,train_result,features, 'svm')
        
        for feature in features_selected:
            top_features[feature]+=1 
        train_data = np.array(train_data[features_selected])
        
        test_data = np.array(test_data[features_selected])
        
        for neighbor in neighbors:
            knn=KNeighborsClassifier(n_neighbors=neighbor)
            knn.fit(train_data,train_result)
            predicted_label = knn.predict(test_data)
            tn, fp, fn, tp = confusion_matrix(test_result, predicted_label).ravel()
            convert_matrix = [tn,fp,fn,tp]
            get_matrix = dict.get(name)
            result = [convert_matrix[i]+get_matrix[i] for i in range(len(get_matrix))]
            dict.update({name: result})
            
            
    
        
    
    

        




def svm(data, column, features, method, file_to_write):
    skf = StratifiedKFold(n_splits=10)
    X = np.array(data.drop(columns = [column], axis = 1))
    y = np.array(data[column])
    top_features = {}
    
    Cs = [0.1, 0.3,0.5,0.7,0.9,1,4,7,10,100]
    gammas = [0.1,0.3,0.5,0.7,0.9,1,4,7,10,100] 
    degrees = [2,3,4,5]
    kernels =['poly', 'linear', 'rbf']
    dict_poly = {}
    dict_linear = {}
    dict_rbf = {}
    
    for c in Cs:
        for gamma in gammas:
            name = str(c)+','+str(gamma)
            dict_linear.update({name:[0,0,0,0]})
            dict_rbf.update({name:[0,0,0,0]})
            for degree in degrees:
                new_name = name+','+str(degree)
                dict_poly.update({new_name:[0,0,0,0]})
    

     
    columns = data.drop(columns = [column], axis = 1).columns 
    for column in columns:
        top_features.update({column:0})
        
    for train, test in skf.split(X,y):
        train_data, test_data = X[train], X[test]  
        train_data, test_data = impute(train_data, test_data)
        train_result, test_result = y[train], y[test] 
        train_data = pd.DataFrame(data = train_data, columns=columns)
        test_data = pd.DataFrame(data = test_data, columns=columns)
        features_selected = run_feature_selection(method,train_data,train_result,features, 'svm')
        
        for feature in features_selected:
            top_features[feature]+=1 
        train_data = np.array(train_data[features_selected])
        
        test_data = np.array(test_data[features_selected])
        
        
        for c in Cs:
            for gamma in gammas:
                name = str(c)+','+str(gamma)
                linear = SVC(probability=True,kernel = 'linear', C=c, gamma=gamma)
                linear.fit(train_data,train_result)
                predicted_label = linear.predict(test_data)
                tn, fp, fn, tp = confusion_matrix(test_result, predicted_label).ravel()
                convert_matrix = [tn,fp,fn,tp]
                get_matrix = dict_linear.get(name)
                result = [convert_matrix[i]+get_matrix[i] for i in range(len(get_matrix))]
                dict_linear.update({name: result})
                
                rbf = SVC(probability=True,kernel = 'rbf', C=c, gamma=gamma)
                rbf.fit(train_data,train_result)
                predicted_label = rbf.predict(test_data)
                tn, fp, fn, tp = confusion_matrix(test_result, predicted_label).ravel()
                convert_matrix = [tn,fp,fn,tp]
                get_matrix = dict_rbf.get(name)
                result = [convert_matrix[i]+get_matrix[i] for i in range(len(get_matrix))]
                dict_rbf.update({name: result})
                
                for degree in degrees:
                    new_name = name+','+str(degree)
                    poly = SVC(probability=True,kernel = 'poly', C=c, gamma=gamma, degree=degree)
                    poly.fit(train_data,train_result)
                    predicted_label = poly.predict(test_data)
                    tn, fp, fn, tp = confusion_matrix(test_result, predicted_label).ravel()
                    convert_matrix = [tn,fp,fn,tp]
                    get_matrix = dict_poly.get(new_name)
                    result = [convert_matrix[i]+get_matrix[i] for i in range(len(get_matrix))]
                    dict_rbf.update({new_name: result})
                              
  
        
        
        
#########Random Forest Classification #################        
def rdforest(data, column, features, method, file_to_write):
    skf = StratifiedKFold(n_splits=10)
    X = np.array(data.drop(columns = [column], axis = 1))
    y = np.array(data[column])
    top_features = {}
    dict = {}
    estimators = [100, 200,300, 400, 500]
    max_features = [5,10]
    for estimator in estimators:
        for max_feature in max_features:
            name = str(estimator)+','+str(max_feature)
            dict.update({name:[0,0,0,0]})
    columns = data.drop(columns = [column], axis = 1).columns 
    for column in columns:
        top_features.update({column:0})
        
        
    for train, test in skf.split(X,y):
        train_data, test_data = X[train], X[test]
        train_data, test_data = impute(train_data, test_data)
        train_result, test_result = y[train], y[test]  
        train_data = pd.DataFrame(data = train_data, columns=columns)
        test_data = pd.DataFrame(data = test_data, columns=columns)
        features_selected = run_feature_selection(method,train_data,train_result,features, 'rdforest')
        for feature in features_selected:
            top_features[feature]+=1
        train_data = np.array(train_data[features_selected])
        
        test_data = np.array(test_data[features_selected])
       
       
        
        for estimator in estimators:
            for max_feature in max_features:
                name = str(estimator)+','+str(max_feature)
                classifier = RandomForestClassifier(n_estimators=estimator, max_features=max_feature,  random_state=0)
                classifier.fit(train_data, train_result)   
                predicted_label = boosted.predict(test_data)
                tn, fp, fn, tp = confusion_matrix(test_result, predicted_label).ravel()
                convert_matrix = [tn,fp,fn,tp]
                get_matrix = dict.get(name)
                result = [convert_matrix[i]+get_matrix[i] for i in range(len(get_matrix))]
                dict.update({name: result})
    return dict
                   
            
    # writer = pd.ExcelWriter(file_to_write)
    # df = pd.DataFrame([dict],index = ['Accuracy'])
    # df.to_excel(writer,'Accuracy')
    # columns = [k for k in top_features]
    # values = [v for v in top_features.values()]
    # names_scores = list(zip( columns, values))
    # ns_df = pd.DataFrame(data = names_scores, columns=['feature', 'scores'])
    # #Sort the dataframe for better visualization
    # ns_df_sorted = ns_df.sort_values( by = ['scores'], ascending = False)
    # ns_df_sorted.to_excel(writer,'Features')
    
    # writer.save()  
    
    
    
    
    
    
# ###########  ElasticNetCV  ##########      
        
def elasticNet (data, column, features, method, file_to_write):
    skf = StratifiedKFold(n_splits=10)
    X = np.array(data.drop(columns = [column], axis = 1))
    y = np.array(data[column])
    top_features = {}
    alphas= np.logspace(-5,5,10)
    l1s = np.linspace(0,1,11)
    dict = {}
    
    for alpha in alphas:
        for l1 in l1s:
            name = str(alpha)+','+str(l1)
            dict.update({name:[0,0,0,0]})
            
  
    
    
  
    columns = data.drop(columns = [column], axis = 1).columns 
    for column in columns:
        top_features.update({column:0})
   
    
    
    
    for train, test in skf.split(X,y):
        
        train_data, test_data = X[train], X[test]
        train_data, test_data = impute(train_data, test_data)
        train_result, test_result = y[train], y[test]  
        train_data = pd.DataFrame(data = train_data, columns=columns)
        test_data = pd.DataFrame(data = test_data, columns=columns)
        features_selected = run_feature_selection(method,train_data,train_result,features, 'elasticNet')
        
        for feature in features_selected:
            top_features[feature]+=1
        train_data = np.array(train_data[features_selected])
        
        test_data = np.array(test_data[features_selected])
        
        for alpha in alphas:
            for l1 in l1s:
                name = str(alpha)+','+str(l1)
                regr = ElasticNet(alpha = alpha,l1_ratio = l1,random_state=0)
        
                model = regr.fit(train_data, train_result)
                predicted_label = model.predict(test_data)>0.5
              
                tn, fp, fn, tp = confusion_matrix(test_result, predicted_label).ravel()
                convert_matrix = [tn,fp,fn,tp]
                get_matrix = dict.get(name)
                result = [convert_matrix[i]+get_matrix[i] for i in range(len(get_matrix))]
                dict.update({name: result})
  
    return dict            
                # name = str(alpha)+','+str(l1)
                # dict[name] = dict[name]+ metrics.accuracy_score(test_result,predicted_label)/10
    
    # writer = pd.ExcelWriter(file_to_write)
    # df = pd.DataFrame([dict],index = ['Accuracy'])
    # df.to_excel(writer,'Accuracy')
    # columns = [k for k in top_features]
    # values = [v for v in top_features.values()]
    # names_scores = list(zip( columns, values))
    # ns_df = pd.DataFrame(data = names_scores, columns=['feature', 'scores'])
    # #Sort the dataframe for better visualization
    # ns_df_sorted = ns_df.sort_values( by = ['scores'], ascending = False)
    # ns_df_sorted.to_excel(writer,'Features')
    
    # writer.save() 
    
    
                    

       

        
                            
       
                
                
    
    
    
    
    
# ########### decision tree classifier ##################
def xgboost(data, column, features, method, file_to_write):   
    skf = StratifiedKFold(n_splits=10,shuffle = True, random_state = 10 ) 
    X = np.array(data.drop(columns = [column], axis = 1))
    y = np.array(data[column])
    top_features = {}
    dict = {} 
    
    columns = data.drop(columns = [column],axis = 1).columns
    for column in columns:
        top_features.update({column:0})
                
    for train, test in skf.split(X,y):   
        train_data, test_data = X[train], X[test]
        train_data, test_data = impute(train_data, test_data)
        train_result, test_result = y[train], y[test] 
        train_data = pd.DataFrame(data = train_data, columns=columns)
        test_data = pd.DataFrame(data = test_data, columns=columns) 
        features_selected = run_feature_selection(method,train_data,train_result,features, 'xgboost')
        
        for feature in features_selected:
            top_features[feature]+=1
            
        
        train_data = np.array(train_data[features_selected])
        
        test_data = np.array(test_data[features_selected])
        
        scoring = ['f1']

        # XGBoost space
        params_xgb = {'learning_rate': np.linspace(0, 1, 6),
                    'min_split_loss': np.linspace(0, 1, 6),
                    'max_depth': np.linspace(2, 10, 5, dtype=int),
                    'min_child_weight': np.linspace(1, 19, 10, dtype=int),
                    'colsample_bytree': np.linspace(0.5, 1, 6),
                    'reg_alpha': np.linspace(0, 5, 6),
                    'scale_pos_weight': np.linspace(2, 10,5, dtype=int),
                    'n_estimators': np.linspace(50, 400, 8, dtype=int),
                    'reg_lambda': np.linspace(0, 5, 6)
                    }

        xgb= RandomizedSearchCV(estimator=XGBClassifier(booster='gbtree',
                                                                            objective='binary:logistic',
                                                                            
                                                                            nthread=1),
                                                    param_distributions=params_xgb,
                                                    n_iter=250,
                                                    scoring=scoring,
                                                    
                                                
                                                    n_jobs=-1,
                                                   
                                                    verbose=1,
                                                    refit='f1')

        search = xgb.fit(train_data, train_result)
        predicted_label = search.predict(test_data)
        tn, fp, fn, tp = confusion_matrix(test_result, predicted_label).ravel()
        convert_matrix = [tn,fp,fn,tp]
        params = search.best_params_

        dict.update({'contingency': convert_matrix, 'params': params})
        return dict
        
        
    #     dict['score']+=metrics.accuracy_score(test_result,ypred)/10
    # writer = pd.ExcelWriter(file_to_write)
    # df = pd.DataFrame([dict],index = ['Accuracy'])
    # df.to_excel(writer,'Accuracy')
    # columns = [k for k in top_features]
    # values = [v for v in top_features.values()]
    # names_scores = list(zip( columns, values))
    # ns_df = pd.DataFrame(data = names_scores, columns=['feature', 'scores'])
    # #Sort the dataframe for better visualization
    # ns_df_sorted = ns_df.sort_values( by = ['scores'], ascending = False)
    # ns_df_sorted.to_excel(writer,'Features')
    
    # writer.save() 
        
    
        

################ naive bayes classifier###############
def naive_bayes(data, column, features, method, file_to_write):
    skf = StratifiedKFold(n_splits=10) 
    X = np.array(data.drop(columns = [column], axis = 1))
    y = np.array(data[column])
    top_features = {}
    dict_Gauss = {}
    dict_Bernoulli = {}
    
    
    
    
    columns = data.drop(columns = [column], axis = 1).columns 
    for column in columns:
        top_features.update({column:0})
        
    
    for train, test in skf.split(X,y):
        
        train_data, test_data = X[train], X[test]
        train_data, test_data = impute(train_data, test_data)
        train_result, test_result = y[train], y[test]  
        train_data = pd.DataFrame(data = train_data, columns=columns)
        test_data = pd.DataFrame(data = test_data, columns=columns)
        features_selected = run_feature_selection(method,train_data,train_result,features, 'naive_bayes')
       
        
        for feature in features_selected:
            top_features[feature]+=1
            
        
        train_data = np.array(train_data[features_selected])
        
        test_data = np.array(test_data[features_selected])
        
        
        
        gnb = GaussianNB()
        predicted_label = gnb.fit(train_data, train_result).predict(test_data)
        tn, fp, fn, tp = confusion_matrix(test_result, predicted_label).ravel()
        convert_matrix = [tn,fp,fn,tp]
        dict_Gauss.update({'contingency': convert_matrix})
        
        
        
        gnb = BernoulliNB()
        predicted_label = gnb.fit(train_data, train_result).predict(test_data)
        tn, fp, fn, tp = confusion_matrix(test_result, predicted_label).ravel()
        convert_matrix = [tn,fp,fn,tp]
        dict_Bernoulli.update({'contingency': convert_matrix})
        
        
       
    # writer = pd.ExcelWriter(file_to_write)
    # df = pd.DataFrame([dict],index = ['Accuracy'])
    # df.to_excel(writer,'Accuracy')
    # columns = [k for k in top_features]
    # values = [v for v in top_features.values()]
    # names_scores = list(zip( columns, values))
    # ns_df = pd.DataFrame(data = names_scores, columns=['feature', 'scores'])
    # #Sort the dataframe for better visualization
    # ns_df_sorted = ns_df.sort_values( by = ['scores'], ascending = False)
    # ns_df_sorted.to_excel(writer,'Features')
    
    # writer.save() 
    
    
def 
    

        

    
    

                
                
            
        
        
        
        
        
        
     







    
    
                    
                
                
        
    
            
            
            
            
            
                
            
        

    
    
def stats(data, column, file_to_write):
    logit = sm.Logit(data[column], data.drop(columns = [column]), axis =1)
    result = logit.fit()
    
    # fittedvalues = data[:, 2]
    # predict_mean_se  = data[:, 3]
    p_values = result.pvalues.to_frame()
    p_values.columns = [''] * len(p_values.columns)
    
    CI = result.conf_int(alpha = 0.05)
    CI = np.exp(CI)
    CI.columns = [''] * len(CI.columns)
    
    coeff = np.exp(result.params.to_frame())
    coeff.columns = [''] * len(coeff.columns)
    
    file_to_write.write('pvalues:')
    file_to_write.write(p_values.to_string()+'\n')
    
    file_to_write.write('CI 95%:')
    file_to_write.write(CI.to_string()+'\n')
    file_to_write.write('odds ratio:')
    file_to_write.write(coeff.to_string()+'\n')
    
def univariate_stats(data,column, file_to_write):
    file_to_write.write('univariate wise: \n')
    columns = data.drop(columns = [column]).columns
    for col in columns:
        logit = sm.Logit(data[column], data[col], axis =1)
        coeff = np.exp(logit.fit().params.to_frame())
        coeff.columns = [''] * len(coeff.columns)
        
        file_to_write.write(coeff.to_string())
    

    
    
    
    
    
    
    

        
        
        
