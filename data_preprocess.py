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
    data = data[data['HPYLORI'].notna()]
    return data

def recategorize(df):
    df["ADD"].replace({2: 0}, inplace=True)
    print(df.columns)
    df["AnyPars3"].replace({2: 0}, inplace=True)
    df["AnyPars3"].replace({9: 0}, inplace=True)
    
    df["CookArea"]=df["CookArea"]-1
    df["GCOW"]=df["GCOW"]-1
   
    
    
    
    df["DEWORM"].replace({0: 'a'}, inplace=True)
    df["DEWORM"].replace({1: 0}, inplace=True)
    df["DEWORM"].replace({'a': 1}, inplace=True)
    
    df["GCAT"].replace({1: 0}, inplace=True)
    df["GCAT"].replace({2: 1}, inplace=True)
    df["GCAT"].replace({3: 2}, inplace=True)
    
    df["GDOG"].replace({1: 0}, inplace=True)
    df["GDOG"].replace({2: 1}, inplace=True)
    df["GDOG"].replace({3: 2}, inplace=True)
    
    df["GELEC"].replace({3: 0}, inplace=True)
    df["GELEC"].replace({2: 'a'}, inplace=True)
    df["GELEC"].replace({1: 2}, inplace=True)
    df["GELEC"].replace({'a':1}, inplace=True)
    
    df["GFLOOR6A"].replace({1: 0}, inplace=True)
    df["GFLOOR6A"].replace({2: 1}, inplace=True)
    df["GFLOOR6A"].replace({3: 2}, inplace=True)
    
    df["GWASTE"].replace({4: 0}, inplace=True)
    
    df["HCIGR6A"].replace({1: 0}, inplace=True)
    df["HCIGR6A"].replace({2: 1}, inplace=True)
    
    df["HPYLORI"].replace({1: 0}, inplace=True)
    df["HPYLORI"].replace({2: 1}, inplace=True)
    
    df["AgeGroups"].replace({1: 0}, inplace=True)
    df["AgeGroups"].replace({2: 1}, inplace=True)
    df["AgeGroups"].replace({3: 2}, inplace=True)
    
    df["FamilyGrouped"].replace({1: 0}, inplace=True)
    df["FamilyGrouped"].replace({2: 1}, inplace=True)
    df["FamilyGrouped"].replace({3: 2}, inplace=True)
    
    df["ToiletType"].replace({1: 0}, inplace=True)
    df["ToiletType"].replace({2: 1}, inplace=True)
    df["ToiletType"].replace({3: 2}, inplace=True)
    
    df["WaterSource"].replace({1: 0}, inplace=True)
    df["WaterSource"].replace({2: 1}, inplace=True)
    df["WaterSource"].replace({3: 2}, inplace=True)
    return df
    
    
def modify_data(data, numerical, nominal):
    
    data = remove_invalid_column(data)
    data = recategorize(data)
    columns = data.columns
    data.columns = columns   
    nominal = getnominal(data)  
  
    nominal_data = create_dummies (data[nominal])
    data = data.drop(nominal, axis = 1)
    data = pd.concat([data,nominal_data], axis =1)
   
    
    return data
    


    
def getnominal(data):
    nominal = []
    for col in data.columns:
        
        distinct = np.sort(data[col].dropna().unique())
        
        if len(distinct) > 2:
            nominal.append(col)
            
        data[col] = data[col] - min(distinct)
   
    return nominal
            
        
        
    
    
    
    
def create_dummies (data):
     
    
   
    dummy = pd.get_dummies(data, columns = data.columns, drop_first= True) 
    return dummy
      
# def normalize(data, numerical):
#     data = preprocessing.StandardScaler().fit_transform(data[numerical])
def impute(data):
    columns = data.columns
    imputed_data = []
    for i in range(5):
        imputer = IterativeImputer(sample_posterior=True, random_state=i, verbose=1)
        imputed_data.append(imputer.fit_transform(data))
    returned_data = np.round(np.mean(imputed_data,axis = 0))
    return_data = pd.DataFrame(data = returned_data, columns=columns)
        
    
    

    return return_data
    
    
    
    
        
    
    

    
    
    
    
        
        
        
    
    
            

    