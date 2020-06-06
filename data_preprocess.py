import pandas as pd
import numpy as np
import math
import csv
import sklearn
import random
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
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
##########################################
def remove_invalid_column(data):
    data = data[data.columns[data.isnull().mean()<0.05]] #excluding columns with too many missing data
    data = data.select_dtypes(exclude=['object']) #excluding columns with wrong data type
    return data
def modify_data(data, numerical, nominal):
    data = remove_invalid_column(data)
    columns = data.columns
    data = pd.DataFrame(impute(data))
    data.columns = columns   
    nominal = getnominal(data)  
    nominal_data = create_dummies (data[nominal])
    data = data.drop(nominal, axis = 1)
    data = pd.concat([data,nominal_data], axis =1)
    
    # normalize(data, numerical)
    return data
    


    
def getnominal(data):
    nominal = []
    for col in data.columns:
        distinct = np.sort(data[col].unique())
        if len(distinct) > 2:
            nominal.append(col)
            
        data[col] = data[col] - min(distinct)
   
    return nominal
            
        
        
    
    
    
    
def create_dummies (data):
    dummy = pd.get_dummies(data, columns = data.columns, drop_first= True) 
    return dummy
      
def normalize(data, numerical):
    data = preprocessing.StandardScaler().fit_transform(data[numerical])
def impute(data):
    imp = IterativeImputer(max_iter = 5, random_state = 0)
    imp.fit(data)
    data = np.round(imp.transform(data))
    return data
    
    
    
    
        
    
    

    
    
    
    
        
        
        
    
    
            

    