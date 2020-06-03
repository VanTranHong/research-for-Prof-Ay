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

from ReliefF import ReliefF
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from mlxtend.feature_selection import SequentialFeatureSelector
from mlxtend.feature_selection import ExhaustiveFeatureSelector
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB






###################feature selection info gain##################


def infogain(train_data, train_result, features):  
    columns = train_data.columns
    X = np.array(train_data)
    y = np.array(train_result)
    mut_info_score = mutual_info_classif(X, y)
    
    
    names_scores = list(zip( train_data.columns, mut_info_score))
    ns_df = pd.DataFrame(data = names_scores, columns=['feature', 'mut_info_score'])
    #Sort the dataframe for better visualization
    ns_df_sorted = ns_df.sort_values( by = ['mut_info_score'], ascending = False)
  
    return ns_df_sorted.iloc[:features,0]
    
def run_feature_selection(method,train_data,train_result,features,classifier):
    if method == 'infogain':
        return infogain(train_data,train_result,features)
    elif method == 'reliefF':
        return reliefF(train_data,train_result,features)
    else:
        return sfs(train_data,train_result,features, classifier)
        
        
        

   
  

        
        

     
    
    
    
    
####################### feature selection ReliefF ################
###when we use reliefF, we only do feature selection for the test set, therefore, before conducting ReliefF, need to split data set
# it will select the subset of k features that are the most predictive of the result

def reliefF(train_data, train_result, features):  
    columns = train_data.columns
    X = np.array(train_data)
    y = np.array(train_result)
    
    # fs.fit_transform(X, y)
    # X_test_subset = fs.transform(X)
    # print(X_test_subset)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    fs = ReliefF(n_neighbors=100, n_features_to_keep=features)
    X_train = fs.fit_transform(X, y)
    index = fs.top_features[:features]
    return [columns[i] for i in index]
    



# ############### feature selection using SequentialFeatureSekector,a Subset Method using python #############
# X_train, X_test, y_train, y_test = train_test_split(data.drop(['Hpylori']),data['Hpylori'], test_size = 0.1, stratify = data['Hpylori'], random_state = random_seed)   
# feature_selector = SequentialFeatureSelector(RandomForestClassifier(n_jobs=-1),
#            k_features=15,
#            forward=True,
#            verbose=2,
#            scoring='roc_auc',
#            cv=5)
# features = feature_selector.fit(X_train, y_train)
# filtered_features= train_features.columns[list(features.k_feature_idx_)]

# ################ feature selection (subset method) using exhaustive feature selection in python ############
# X_train, X_test, y_train, y_test = train_test_split(data.drop(['Hpylori']),data['Hpylori'], test_size = 0.1, stratify = data['Hpylori'], random_state = random_seed)   
# feature_selector = ExhaustiveFeatureSelector(RandomForestClassifier(n_jobs=-1),
#            min_features=10,
#            max_features=20,
#            scoring='roc_auc',
#            print_progress=True,
#            cv=5)
# features = feature_selector.fit(X_train, y_train)
# filtered_features= train_features.columns[list(features.k_feature_idx_)]

# ################## feature selection using SFS ##############
def sfs(train_data, train_result, features, classifier):  
    columns = train_data.columns
    X = np.array(train_data).astype(float)
    y = np.array(train_result).astype(float)
    if classifier == 'svm':
        classifier = SVC()
    elif classifier == 'rdforest':
        classifier = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    elif classifier =='lasso':
        classifier = Lasso(alpha=0.5)
    elif classifier == 'elasticNet':
        classifier = ElasticNet()
    elif classifier == 'xgboost':
        classifier = XGBClassifier()
    else:
        classifier = BernoulliNB(binarize=0.0)
        
    
    sfs = SFS(classifier, k_features=features, forward=True, floating=True, scoring='f1', verbose=1, cv=0, n_jobs=-1)
    print(X)
    print(y)
    sfs.fit(X, y)
    # index = list(sfs.k_feature_names_)
    # index = [int(s) for s in index]
   

    
    # return [columns[i] for i in index]
    
    
    
    
####### finding some other subset based method that is independent of the classifier
##### classifier dependent and classifier independent

    
 
    
   
   

