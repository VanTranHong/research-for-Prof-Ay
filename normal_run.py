import pandas as pd
import numpy as np
import math
import csv
import sklearn
import random
from openpyxl import Workbook, load_workbook
from openpyxl.drawing.image import Image
from openpyxl.utils.dataframe import dataframe_to_rows
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Lasso
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
from sklearn.model_selection import StratifiedKFold


##importing relevant fuctions from other files#####
from classifiers import svm,rdforest, elasticNet,xgboost, naive_bayes,stats, univariate_stats, KNN
from data_preprocess import modify_data, impute
from CFS import CFS
from featureselection import infogain, reliefF,run_feature_selection,sfs




masterkey = {}
runs = ['infogain,10','infogain,20','reliefF,10', 'reliefF20','infogain,data.shape[1]-1', 'CFS,data.shape[1]-1']
for name in runs:
    insidedict={}
    masterkey.update({name:insidedict})
    
    
    
def run(data, column):
    for run in runs:
        elasticnet_,xgboost_,svm_linear_,svm_poly_,svm_rbf_,rdforest_,knn_,nb_gauss_,nb_bernoulli_,features_= normal_run(run,data, column)
        insidedict = masterkey.get(run)
        insidedict.update({'elasticnet':elasticnet_,'xgboost':xgboost_,'svm_linear':svm_linear_, 'svm_poly': svm_poly_,'svm_rbf':svm_rbf_,'rdforest':rdforest_,'knn':knn_,'nb_gauss':nb_gauss_,'nb_bernoulli':nb_bernoulli_,'features':features_ })
    return masterkey

def run_sfs(data, column):
    features = data.shape[1]-1
    feature_selection = 'sfs'
    master = {}
    for metric in ['f1', 'accuracy','precision','recall','MMC']:
        innerdict =  {}
        elasticnet_ = []
        xgboost_=[]
        svm_linear_=[]
        svm_poly_=[]
        svm_rbf_=[]
        rdforest_=[]
        knn_=[]
        nb_gauss_ =[]
        nb_bernoulli_= []
        
        features_ = []
        
        for i in range(10):
            dict,top_features = elasticNet(data,column, features, feature_selection, metric)
            elasticnet_.append(dict)
            features_.append(top_features)
            
            dict, top_features = xgboost(data,column, features, feature_selection, metric)
            xgboost_.append(dict)
            features_.append(top_features)
            
            dict, top_features = KNN(data,column, features, feature_selection, metric)
            knn_.append(dict)
            features_.append(top_features)
            
            linear, poly, rbf, top_features = svm(data,column, features, feature_selection, metric)
            svm_linear_.append(linear)
            svm_poly_.append(poly)
            svm_rbf_.append(rbf)
            features_.append(top_features)
            
            dict,top_features = rdforest(data,column, features, feature_selection, metric)
            rdforest_.append(dict)
            features_.append(top_features)
            
            gauss, bernoulli, top_features = naive_bayes(data,column, features, feature_selection, metric)
            nb_gauss_.append(gauss)
            nb_bernoulli_.append(bernoulli)
            features_.append(top_features)
        innerdict.update({'elasticnet':elasticnet_,'xgboost':xgboost_,'svm_linear':svm_linear_, 'svm_poly': svm_poly_,'svm_rbf':svm_rbf_,'rdforest':rdforest_,'knn':knn_,'nb_gauss':nb_gauss_,'nb_bernoulli':nb_bernoulli_,'features':features_ })
        master.update({metric:innerdict})
    return master
    
        
        
        
        
        
        
    
        



def normal_run(run, data, column):
    params = run.split(',')
    feature_selection = params[0]
    features = int(params[1])
    
    
    elasticnet_ = []
    xgboost_=[]
    svm_linear_=[]
    svm_poly_=[]
    svm_rbf_=[]
    rdforest_=[]
    knn_=[]
    nb_gauss_ =[]
    nb_bernoulli_= []
    
    features_ = []
    
    for i in range(10):
        dict,top_features = elasticNet(data,column, features, feature_selection, metric)
        elasticnet_.append(dict)
        features_.append(top_features)
        
        dict, top_features = xgboost(data,column, features, feature_selection, metric)
        xgboost_.append(dict)
        features_.append(top_features)
        
        dict, top_features = KNN(data,column, features, feature_selection, metric)
        knn_.append(dict)
        features_.append(top_features)
        
        linear, poly, rbf, top_features = svm(data,column, features, feature_selection, metric)
        svm_linear_.append(linear)
        svm_poly_.append(poly)
        svm_rbf_.append(rbf)
        features_.append(top_features)
        
        dict,top_features = rdforest(data,column, features, feature_selection, metric)
        rdforest_.append(dict)
        features_.append(top_features)
        
        gauss, bernoulli, top_features = naive_bayes(data,column, features, feature_selection, metric)
        nb_gauss_.append(gauss)
        nb_bernoulli_.append(bernoulli)
        features_.append(top_features)
    return elasticnet_,xgboost_,svm_linear_,svm_poly_,svm_rbf_,rdforest_,knn_,nb_gauss_,nb_bernoulli_,features_
            
        
    
    