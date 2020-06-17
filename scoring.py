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
    
def get_accuracy(contingency): 
    return (contingency[0]+contingency[3])/(contingency[0]+contingency[1]+contingency[2]+contingency[3])
        
def f1(dict, classfier):
    keys = dict.keys()
    f1_dict= {}
    for key in keys:
        contingency = dict.get(key)
        f1 = get_f1(contingency)
        f1_dict.update({key:f1})
    Keymax = max(f1_dict, key=f1_dict.get) 

    return max(f1_dict.values()), Keymax, dict.get(Keymax)
        
    
    
    
    
    
def accuracy(dict):
    keys = dict.keys()
    accuracy_dict= {}
    for key in keys:
        contingency = dict.get(key)
        accuracy = get_accuracy(contingency)
        accuracy_dict.update({key:accuracy})
    Keymax = max(accuracy_dict, key=accuracy_dict.get) 

    return max(accuracy_dict.values()), Keymax, dict.get(Keymax)
    
    
def precision(dict):
    
    keys = dict.keys()
    precision_dict= {}
    for key in keys:
        contingency = dict.get(key)
        precision = get_precision(contingency)
        precision_dict.update({key:precision})
    Keymax = max(precision_dict, key=precision_dict.get) 

    return max(precision_dict.values()), Keymax, dict.get(Keymax)
    
    
def PPV(dict):
    keys = dict.keys()
    PPV_dict= {}
    for key in keys:
        contingency = dict.get(key)
        PPV = get_PPV(contingency)
        PPV_dict.update({key:PPV})
    Keymax = max(PPV_dict, key=PPV_dict.get) 

    return max(PPV_dict.values()), Keymax, dict.get(Keymax)
    