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

def bag(classifier,n_est):
    abc = BaggingClassifier(base_estimator=classifier,n_estimators=n_est, random_state=0)
    return abc

def run_bagging(data, column, features, method, parameters):
    masterdict = {}
    svm_linear_ = bag_svm(data, column, features, method,parameters[0],'linear')
    svm_poly_ = bag_svm(data, column, features, method,parameters[1],'poly')
    svm_rbf_ =bag_svm(data, column, features, method,parameters[2],'rbf')
    rdforest_ = bag_rdforest(data, column, features, method,parameters[3])
    knn_ = bag_knn(data, column, features, method,parameters[4])
    nb_gauss_ = bag_naive_bayes(data, column, features, method,parameters[5],'Gaussian')
    nb_bernoulli_ = bag_naive_bayes(data, column, features, method,parameters[5],'Bernoulli')
    
    masterdict.update({'svm_linear':svm_linear_, 'svm_poly': svm_poly_,'svm_rbf':svm_rbf_,'rdforest':rdforest_,'knn':knn_,'nb_gauss':nb_gauss_,'nb_bernoulli':nb_bernoulli_})
    return masterdict
    


def bag_svm(data, column, features, method, params, kernel):
    skf = StratifiedKFold(n_splits=10)
    X = np.array(data.drop(columns = [column], axis = 1))
    y = np.array(data[column])
    top_features = {}
    dict = {}
    n_ests = [25,50,75,100]
   
    for n_est in n_ests:
        
        name = str(n_est)
        dict.update({name:[0,0,0,0]})
    
    
    
    if kernel =='rbf' or 'linear':
        parameters = params.split(',')
        c= float(parameters[0])
        gamma = float(parameters[1])
        
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
            for n_est in n_ests:
                
                name = str(n_est)
                estimator = SVC(probability=True,kernel = kernel, C=c, gamma=gamma)
                bag = bag(estimator, n_est)  
                bag.fit(train_data,train_result)
                predicted_label = bag.predict(test_data)
                tn, fp, fn, tp = confusion_matrix(test_result, predicted_label).ravel()
                convert_matrix = [tn,fp,fn,tp]
                get_matrix = dict.get(name)
                result = [convert_matrix[i]+get_matrix[i] for i in range(len(get_matrix))]
                dict.update({name: result})
        
    
    else:
        parameters = params.split(',')
        c= float(parameters[0])
        gamma = float(parameters[1])
        degree = int(parameters[2])
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
            for n_est in n_ests:
                
                name = str(n_est)
                estimator = SVC(probability=True,kernel = kernel, C=c, gamma=gamma, degree = degree)
                bag = bag(estimator, n_est)  
                bag.fit(train_data,train_result)
                predicted_label = bag.predict(test_data)
                tn, fp, fn, tp = confusion_matrix(test_result, predicted_label).ravel()
                convert_matrix = [tn,fp,fn,tp]
                get_matrix = dict.get(name)
                result = [convert_matrix[i]+get_matrix[i] for i in range(len(get_matrix))]
                dict.update({name: result})
    return dict
        
        
    
    
    
    
def bag_rdforest(data, column, features, method, params):
    skf = StratifiedKFold(n_splits=10)
    X = np.array(data.drop(columns = [column], axis = 1))
    y = np.array(data[column])
    top_features = {}
    dict = {}
    n_ests = [25,50,75,100]
   
    for n_est in n_ests:
        
        name = str(n_est)
        dict.update({name:[0,0,0,0]})
    parameters = params.split(',')
    estimator= int(parameters[0])
    max_feature = int(parameters[1])
        
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
        for n_est in n_ests:
                
                name = str(n_est)
                estimator = RandomForestClassifier(n_estimators=estimator, max_features=max_feature)
                bag = bag(estimator, n_est)  
                bag.fit(train_data,train_result)
                predicted_label = bag.predict(test_data)
                tn, fp, fn, tp = confusion_matrix(test_result, predicted_label).ravel()
                convert_matrix = [tn,fp,fn,tp]
                get_matrix = dict.get(name)
                result = [convert_matrix[i]+get_matrix[i] for i in range(len(get_matrix))]
                dict.update({name: result})
                # estimator = RandomForestClassifier(n_estimators=estimator, max_features=max_feature)
    return dict
    
def bag_naive_bayes(data, column, features, method, params, kernel):
    skf = StratifiedKFold(n_splits=10)
    X = np.array(data.drop(columns = [column], axis = 1))
    y = np.array(data[column])
    top_features = {}
    dict = {}
   
    
    n_ests = [25,50,75,100]
   
    for n_est in n_ests:
        
        name = str(n_est)
        dict_Gauss.update({name:[0,0,0,0]})
        dict_Bernoulli.update({name:[0,0,0,0]})
    
    
    
    
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
        for n_est in n_ests:
            
            name = str(n_est)
            if kernel=='Gaussian'L
                
                estimator = GaussianNB()
                bag = bag(estimator, n_est)  
                bag.fit(train_data,train_result)
                predicted_label = bag.predict(test_data)
                tn, fp, fn, tp = confusion_matrix(test_result, predicted_label).ravel()
                convert_matrix = [tn,fp,fn,tp]
                get_matrix = dict.get(name)
                result = [convert_matrix[i]+get_matrix[i] for i in range(len(get_matrix))]
                dict.update({name: result})
            else:
                
                estimator = BernoulliNB()
                bag = bag(estimator, n_est)  
                bag.fit(train_data,train_result)
                predicted_label = bag.predict(test_data)
                tn, fp, fn, tp = confusion_matrix(test_result, predicted_label).ravel()
                convert_matrix = [tn,fp,fn,tp]
                get_matrix = dict.get(name)
                result = [convert_matrix[i]+get_matrix[i] for i in range(len(get_matrix))]
                dict.update({name: result})
    return dict
    
            
def bag_knn(data, column, features, method, params):
    skf = StratifiedKFold(n_splits=10)
    X = np.array(data.drop(columns = [column], axis = 1))
    y = np.array(data[column])
    top_features = {}
    dict={}
    n_ests = [25,50,75,100]
    for n_est in n_ests:
        name = str(n_est)
        dict.update({name:[0,0,0,0]})
        
    
    neighbor = int(params)
    
    
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
        for n_est in n_ests:
            
            name = str(n_est)
            estimator = KNeighborsClassifier(n_neighbors=neighbor)
            bag = bag(estimator, n_est)  
            bag.fit(train_data,train_result)
            predicted_label = bag.predict(test_data)
            tn, fp, fn, tp = confusion_matrix(test_result, predicted_label).ravel()
            convert_matrix = [tn,fp,fn,tp]
            get_matrix = dict.get(name)
            result = [convert_matrix[i]+get_matrix[i] for i in range(len(get_matrix))]
            dict.update({name: result})
    return dict
    
    
