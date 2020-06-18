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

# tn,fp,fn,tp
def get_f1(contingency):
    return 2*get_precision(contingency)*get_PPV(contingency)/(get_precision(contingency)+get_PPV(contingency))

def get_precision(contingency):
    return contingency[3]/(contingency[3]+contingency[1])
    
def get_PPV(contingency): # recall
    return contingency[3]/(contingency[3]+contingency[2])
    
def get_MCC(contingency):
    up = contingency[3]*contingency[0]-contingency[1]*contingency[2]
    down = math.sqrt((contingency[1]+contingency[3])*(contingency[2]+contingency[3])*(contingency[1]+contingency[0])*(contingency[0]+contingency[2]))
    return up/down
def get_accuracy(contingency): 
    return (contingency[0]+contingency[3])/(contingency[0]+contingency[1]+contingency[2]+contingency[3])
def MMC(dict):
    keys = dict[0].keys()
    MMC_dict= {}
    for key in keys:
        MMC_dict.update({key:0})
    for i in range(10):
        innerdict = dict[i]
        for key in keys:
        
            contingency = innerdict.get(key)
            mmc = get_MMC(contingency)
            old_value = MMC_dict.get(key)
            MMC_dict.update({key:mmc+old_value})
    Keymax = max(MMC_dict, key=MMC_dict.get) 

    return max(MMC_dict.values())/10, Keymax, dict.get(Keymax)
    
  
  
  
        
def f1(dict): #######dict is a list of dicts
    keys = dict[0].keys()
    f1_dict= {}
    for key in keys:
        f1_dict.update({key:0})
    for i in range(10):
        innerdict = dict[i]
        for key in keys:
        
            contingency = innerdict.get(key)
            f1 = get_f1(contingency)
            old_value = f1_dict.get(key)
            f1_dict.update({key:f1+old_value})
    Keymax = max(f1_dict, key=f1_dict.get) 

    return max(f1_dict.values())/10, Keymax, dict.get(Keymax)
        
    
    
def accuracy(dict): ##### list of dict
    keys = dict[0].keys()
    accuracy_dict= {}
    for key in keys:
        accuracy_dict.update({key:0})
    for i in range(10):
        innerdict = dict[i]
        for key in keys:
        
            contingency = innerdict.get(key)
            accuracy = get_accuracy(contingency)
            old_value = accuracy_dict.get(key)
            accuracy_dict.update({key:accuracy+old_value})
    Keymax = max(accuracy_dict, key=accuracy_dict.get) 

    return max(accuracy_dict.values())/10, Keymax, dict.get(Keymax)
    
    
def precision(dict):
    
    keys = dict[0].keys()
    precision_dict= {}
    for key in keys:
        precision_dict.update({key:0})
    for i in range(10):
        innerdict = dict[i]
        for key in keys:
        
            contingency = innerdict.get(key)
            precision = get_precision(contingency)
            old_value = precision_dict.get(key)
            precision_dict.update({key:precision+old_value})
    Keymax = max(precision_dict, key=precision_dict.get) 

    return max(precision_dict.values())/10, Keymax, dict.get(Keymax)
    
    
def PPV(dict): ####dict is a list of dict of contingency
    
    keys = dict[0].keys()
    PPV_dict= {}
    for key in keys:
        PPV_dict.update({key:0})
    for i in range(10):
        innerdict = dict[i]
        for key in keys:
        
            contingency = innerdict.get(key)
            PPV = get_PPV(contingency)
            old_value = PPV_dict.get(key)
            PPV_dict.update({key:PPV+old_value})
    Keymax = max(PPV_dict, key=PPV_dict.get) 

    return max(PPV_dict.values())/10, Keymax, dict.get(Keymax)

def score(dict_dict):
    master_dict = {}
    methods = dict_dict.keys
    for method in methods: #### for each method elasticNet, xgboost and so on..
        innerdict = {}
        f1_ = {}
        highest, params, contingency = f1(method)
        f1_.update({'value':highest,'params':params,'contingency':contingency})
        
        
        accuracy_ = {}
        highest, params, contingency = accuracy(method)
        accuacy_.update({'value':highest,'params':params,'contingency':contingency})
        
        precision_ = {}
        highest, params, contingency = precision(method)
        precision_.update({'value':highest,'params':params,'contingency':contingency})
        
        PPV_ = {}
        highest, params, contingency = PPV(method)
        PPV_.update({'value':highest,'params':params,'contingency':contingency})
        
        MMC_ = {}
        highest, params, contingency = MMC(method)
        MMC_.update({'value':highest,'params':params,'contingency':contingency})
        
        innerdict.update({'f1':f1_, 'accuracy':accuracy_, 'precision':precision_, 'PPV':PPV_, 'MMC':MMC_})
        master_dict.update({method:innerdict})
    return master_dict
        
        
       
        
        
        
    
    
    
    