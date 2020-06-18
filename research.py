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








####handling the old data

#reading files and removing rows or columns with too many missing places
datafile = r'/Users/vantran/Documents/vantran/info.csv'


#######    main    ####
data = pd.read_csv(datafile,skipinitialspace=True, header = 0)

data = modify_data(data, [],data.columns)

############ producing univariate and multivariet regression #########

# stats_file = open('Stats.txt','w')
# imputed_data = impute(data)
# # print(imputed_data)
# stats(imputed_data,'HPYLORI',stats_file)
# univariate_stats(imputed_data,'HPYLORI',stats_file)
# stats_file.close()


########## baseline accuracy ##########
# f = open('research_result.txt','w')
# rate = sum(data['HPYLORI'])/data.shape[0]
# f.write('base line accuracy is '+str( max(rate,1-rate))+'\n')
# f.close()

############# running 10 times 10-fold cross validation without boosting or bagging for 

############## 10 features infogain ##########
elasticnet_dict_10 = []
xgboost_dict_10=[]
svm_linear_dict_10=[]
svm_poly_dict_10=[]
svm_rbf_dict_10=[]
rdforest_dict_10=[]
knn_dict_10=[]
nb_gauss_dict_10 =[]
nb_bernoulli_dict_10 = []

features_10_infogain =[]

for i in range(10):
    dict,top_features = elasticNet(data,'HPYLORI', 10, 'infogain')
    elasticnet_dict_10.append(dict)
    features_10_infogain.append(top_features)
    
    dict, top_features = xgboost(data,'HPYLORI', 10, 'infogain')
    xgboost_dict_10.append(dict)
    features_10_infogain.append(top_features)
    
    dict, top_features = KNN(data,'HPYLORI', 10, 'infogain')
    knn_dict_10.append(dict)
    features_10_infogain.append(top_features)
    
    linear, poly, rbf, top_features = svm(data,'HPYLORI', 10, 'infogain')
    svm_linear_dict_10.append(linear)
    svm_poly_dict_10.append(poly)
    svm_rbf_dict_10.append(rbf)
    features_10_infogain.append(top_features)
    
    dict,top_features = rdforest(data,'HPYLORI', 10, 'infogain')
    rdforest_dict_10.append(dict)
    features_10_infogain.append(top_features)
    
    gauss, bernoulli, top_features = naive_bayes(data,'HPYLORI', 10, 'infogain')
    nb_gauss_dict_10.append(gauss)
    nb_bernoulli_dict_10.append(bernoulli)
    features_10_infogain.append(top_features)


############## 20 features infogain ###########



features_20_infogain =[]

for i in range(10):
    dict,top_features = elasticNet(data,'HPYLORI', 20, 'infogain')
    elasticnet_dict_20.append(dict)
    features_20_infogain.append(top_features)
    
    dict, top_features = xgboost(data,'HPYLORI', 20, 'infogain')
    xgboost_dict_20.append(dict)
    features_20_infogain.append(top_features)
    
    dict, top_features = KNN(data,'HPYLORI', 20, 'infogain')
    knn_dict_20.append(dict)
    features_20_infogain.append(top_features)
    
    linear, poly, rbf, top_features = svm(data,'HPYLORI', 20, 'infogain')
    svm_linear_dict_20.append(linear)
    svm_poly_dict_20.append(poly)
    svm_rbf_dict_20.append(rbf)
    features_20_infogain.append(top_features)
    
    dict,top_features = rdforest(data,'HPYLORI', 20, 'infogain')
    rdforest_dict_20.append(dict)
    features_20_infogain.append(top_features)
    
    gauss, bernoulli, top_features = naive_bayes(data,'HPYLORI', 20, 'infogain')
    nb_gauss_dict_20.append(gauss)
    nb_bernoulli_dict_20.append(bernoulli)
    features_20_infogain.append(top_features)


########### 10 features reliefF ##############

elasticnet_reliefF_10 = []
xgboost_reliefF_10=[]
svm_linear_reliefF_10=[]
svm_poly_reliefF_10=[]
svm_rbf_reliefF_10=[]
rdforest_reliefF_10=[]
knn_reliefF_10=[]
nb_gauss_reliefF_10 =[]
nb_bernoulli_reliefF_10 = []

features_10_reliefF =[]

for i in range(10):
    dict,top_features = elasticNet(data,'HPYLORI', 10, 'reliefF')
    elasticnet_reliefF_10.append(dict)
    features_10_reliefF .append(top_features)
    
    dict, top_features = xgboost(data,'HPYLORI', 10, 'reliefF')
    xgboost_reliefF_10.append(dict)
    features_10_reliefF .append(top_features)
    
    dict, top_features = KNN(data,'HPYLORI', 10, 'reliefF')
    knn_reliefF_10.append(dict)
    features_10_reliefF .append(top_features)
    
    linear, poly, rbf, top_features = svm(data,'HPYLORI', 10, 'reliefF')
    svm_linear_reliefF_10.append(linear)
    svm_poly_reliefF_10.append(poly)
    svm_rbf_reliefF_10.append(rbf)
    features_10_reliefF .append(top_features)
    
    dict,top_features = rdforest(data,'HPYLORI', 10, 'reliefF')
    rdforest_reliefF_10.append(dict)
    features_10_reliefF .append(top_features)
    
    gauss, bernoulli, top_features = naive_bayes(data,'HPYLORI', 10, 'reliefF')
    nb_gauss_reliefF_10.append(gauss)
    nb_bernoulli_reliefF_10.append(bernoulli)
    features_10_reliefF .append(top_features)


########## 20 features reliefF #############

elasticnet_reliefF_20 = []
xgboost_reliefF_20=[]
svm_linear_reliefF_20=[]
svm_poly_reliefF_20=[]
svm_rbf_reliefF_20=[]
rdforest_reliefF_20=[]
knn_reliefF_20=[]
nb_gauss_reliefF_20 =[]
nb_bernoulli_reliefF_20 = []

features_20_reliefF =[]

for i in range(10):
    dict,top_features = elasticNet(data,'HPYLORI', 20, 'reliefF')
    elasticnet_reliefF_20.append(dict)
    features_20_reliefF .append(top_features)
    
    dict, top_features = xgboost(data,'HPYLORI', 20, 'reliefF')
    xgboost_reliefF_20.append(dict)
    features_20_reliefF .append(top_features)
    
    dict, top_features = KNN(data,'HPYLORI', 20, 'reliefF')
    knn_reliefF_20.append(dict)
    features_20_reliefF .append(top_features)
    
    linear, poly, rbf, top_features = svm(data,'HPYLORI', 20, 'reliefF')
    svm_linear_reliefF_20.append(linear)
    svm_poly_reliefF_20.append(poly)
    svm_rbf_reliefF_20.append(rbf)
    features_20_reliefF .append(top_features)
    
    dict,top_features = rdforest(data,'HPYLORI', 20, 'reliefF')
    rdforest_reliefF_20.append(dict)
    features_20_reliefF .append(top_features)
    
    gauss, bernoulli, top_features = naive_bayes(data,'HPYLORI', 20, 'reliefF')
    nb_gauss_reliefF_20.append(gauss)
    nb_bernoulli_reliefF_20.append(bernoulli)
    features_20_reliefF .append(top_features)


############# full###############

elasticnet_full = []
xgboost_full=[]
svm_linear_full=[]
svm_poly_full=[]
svm_rbf_full=[]
rdforest_full=[]
knn_full=[]
nb_gauss_full =[]
nb_bernoulli_full = []




for i in range(10):
    dict,top_features = elasticNet(data,'HPYLORI', data.shape[1]-1, 'reliefF')
    elasticnet_full.append(dict)
 
    
    dict, top_features = xgboost(data,'HPYLORI', data.shape[1]-1, 'reliefF')
    xgboost_full.append(dict)
    
    
    dict, top_features = KNN(data,'HPYLORI', data.shape[1]-1, 'reliefF')
    knn_full.append(dict)
   
    
    linear, poly, rbf, top_features = svm(data,'HPYLORI', data.shape[1]-1, 'reliefF')
    svm_linear_full.append(linear)
    svm_poly_full.append(poly)
    svm_rbf_full.append(rbf)
  
    
    dict,top_features = rdforest(data,'HPYLORI', data.shape[1]-1, 'reliefF')
    rdforest_full.append(dict)
    features_full .append(top_features)
    
    gauss, bernoulli, top_features = naive_bayes(data,'HPYLORI', data.shape[1]-1, 'reliefF')
    nb_gauss_full.append(gauss)
    nb_bernoulli_full.append(bernoulli)
    features_full.append(top_features)
    
    
    
##################sfs##################



elasticnet_sfs = []
xgboost_sfs =[]
svm_linear_sfs =[]
svm_poly_sfs =[]
svm_rbf_sfs =[]
rdforest_sfs =[]
knn_sfs =[]
nb_gauss_sfs  =[]
nb_bernoulli_sfs  = []

features_sfs = []



for i in range(10):
    dict,top_features = elasticNet(data,'HPYLORI', data.shape[1]-1, 'sfs')
    elasticnet_sfs.append(dict)
    features_sfs.append(top_features)
    
 
    
    dict, top_features = xgboost(data,'HPYLORI', data.shape[1]-1, 'sfs')
    xgboost_sfs.append(dict)
    features_sfs.append(top_features)
    
    
    dict, top_features = KNN(data,'HPYLORI', data.shape[1]-1, 'sfs')
    knn_sfs.append(dict)
    features_sfs.append(top_features)
   
    
    linear, poly, rbf, top_features = svm(data,'HPYLORI', data.shape[1]-1, 'sfs')
    svm_linear_sfs.append(linear)
    svm_poly_sfs.append(poly)
    svm_rbf_sfs.append(rbf)
    features_sfs.append(top_features)
  
    
    dict,top_features = rdforest(data,'HPYLORI', data.shape[1]-1, 'sfs')
    rdforest_sfs.append(dict)
    features_sfs .append(top_features)
    features_sfs.append(top_features)
    
    gauss, bernoulli, top_features = naive_bayes(data,'HPYLORI', data.shape[1]-1, 'sfs')
    nb_gauss_sfs.append(gauss)
    nb_bernoulli_sfs.append(bernoulli)
    features_sfs.append(top_features)
    features_sfs.append(top_features)
    
    
    ################## CFS ##################



elasticnet_CFS = []
xgboost_CFS =[]
svm_linear_CFS =[]
svm_poly_CFS =[]
svm_rbf_CFS =[]
rdforest_CFS =[]
knn_CFS =[]
nb_gauss_CFS  =[]
nb_bernoulli_CFS = []

features_CFS = []



for i in range(10):
    dict,top_features = elasticNet(data,'HPYLORI', data.shape[1]-1, 'CFS')
    elasticnet_CFS.append(dict)
    features_CFS.append(top_features)
    
 
    
    dict, top_features = xgboost(data,'HPYLORI', data.shape[1]-1, 'CFS')
    xgboost_CFS.append(dict)
    features_CFS.append(top_features)
    
    
    dict, top_features = KNN(data,'HPYLORI', data.shape[1]-1, 'CFS')
    knn_CFS.append(dict)
    features_CFS.append(top_features)
   
    
    linear, poly, rbf, top_features = svm(data,'HPYLORI', data.shape[1]-1, 'CFS')
    svm_linear_CFS.append(linear)
    svm_poly_CFS.append(poly)
    svm_rbf_CFS.append(rbf)
    features_CFS.append(top_features)
  
    
    dict,top_features = rdforest(data,'HPYLORI', data.shape[1]-1, 'CFS')
    rdforest_CFS.append(dict)
    features_CFS .append(top_features)
    features_CFS.append(top_features)
    
    gauss, bernoulli, top_features = naive_bayes(data,'HPYLORI', data.shape[1]-1, 'CFS')
    nb_gauss_CFS.append(gauss)
    nb_bernoulli_CFS.append(bernoulli)
    
    features_CFS.append(top_features)
    
    
################ getting the parameters based on f1, accuracy, ppv, precision





elasticnet_10_score = score(elasticnet_dict_10 )
xgboost_10_score =  score(xgboost_dict_10)
svm_linear_10_score = score(svm_linear_dict_10)
svm_poly_10_score =  score(svm_poly_dict_10)
svm_rbf_10_score =  score(svm_rbf_dict_10)
rdforest_10_score =  score(rdforest_dict_10)
knn_10_score =  score(knn_dict_10)
nb_gauss_10_score =  score(nb_gauss_dict_10)
nb_bernoulli_10_score =  score(nb_bernoulli_dict_10)


elasticnet_20_score =  score(elasticnet_dict_20)
xgboost_20_score =  score(xgboost_dict_20)
svm_linear_20_score =  score(svm_linear_dict_20)
svm_poly_20_score =  score(svm_poly_dict_20)
svm_rbf_20_score =  score(svm_rbf_dict_20)
rdforest_20_score =  score(rdforest_dict_20)
knn_20_score =  score(knn_dict_20)
nb_gauss_20_score =  score(nb_gauss_dict_20)
nb_bernoulli_20_score =  score(nb_bernoulli_dict_20)


elasticnet_10_reliefF_score =  score(elasticnet_reliefF_10)
xgboost_10_reliefF_score =  score(xgboost_reliefF_10)
svm_linear_10_reliefF_score =  score(svm_linear_reliefF_10)
svm_poly_10_reliefF_score =  score(svm_poly_reliefF_10)
svm_rbf_10_reliefF_score =  score(svm_rbf_reliefF_10)
rdforest_10_reliefF_score =  score(rdforest_reliefF_10)
knn_10_reliefF_score =  score(knn_reliefF_10)
nb_gauss_10_reliefF_score =  score(nb_gauss_reliefF_10 )
nb_bernoulli_10_reliefF_score =  score(nb_bernoulli_reliefF_10)


elasticnet_20_reliefF_score =  score(elasticnet_reliefF_20)
xgboost_20_reliefF_score =  score(xgboost_reliefF_20)
svm_linear_20_reliefF_score = score(svm_linear_reliefF_20)
svm_poly_20_reliefF_score =  score(svm_poly_reliefF_20)
svm_rbf_20_reliefF_score =  score(svm_rbf_reliefF_20)
rdforest_20_reliefF_score =  score(rdforest_reliefF_20)
knn_20_reliefF_score =  score(knn_reliefF_20)
nb_gauss_20_reliefF_score =  score(nb_gauss_reliefF_20)
nb_bernoulli_20_reliefF_score =  score(nb_bernoulli_reliefF_20)



elasticnet_full_score =  score(elasticnet_full)
xgboost_full_score =  score(xgboost_full)
svm_linear_full_score =  score(svm_linear_full)
svm_poly_full_score =  score(svm_poly_full)
svm_rbf_full_score =  score(svm_rbf_full)
rdforest_full_score =  score(rdforest_full)
knn_full_score =  score(knn_full)
nb_gauss_full_score =  score(nb_gauss_full)
nb_bernoulli_full_score =  score(nb_bernoulli_full )

elasticnet_sfs_score =  score(elasticnet_sfs)
xgboost_sfs_score =  score(xgboost_sfs)
svm_linear_sfs_score =  score(svm_linear_sfs)
svm_poly_sfs_score =  score(svm_poly_sfs)
svm_rbf_sfs_score =  score(svm_rbf_sfs)
rdforest_sfs_score =  score(rdforest_sfs)
knn_sfs_score =  score(knn_sfs)
nb_gauss_sfs_score =  score(nb_gauss_sfs)
nb_bernoulli_sfs_score =  score(nb_bernoulli_sfs )

elasticnet_CFS_score =  score(elasticnet_CFS)
xgboost_CFS_score =  score(xgboost_CFS)
svm_linear_CFS_score =  score(svm_linear_CFS)
svm_poly_CFS_score =  score(svm_poly_CFS)
svm_rbf_CFS_score =  score(svm_rbf_CFS)
rdforest_CFS_score =  score(rdforest_CFS)
knn_CFS_score =  score(knn_CFS)
nb_gauss_CFS_score =  score(nb_gauss_CFS)
nb_bernoulli_CFS_score =  score(nb_bernoulli_CFS)


################ get parameters for boosting and bagging ########
# if want to get the best param based on f1, simply call score = 'f1'

score = 'f1' ####### this is the default, change to get a different metric
def get_param(scores,dict):
    dict_score = dict.get(scores)
    return dict_score.get('params')

elasticnet_10_param = get_param(score,elasticnet_10_score)
xgboost_10_param =  get_param(score,xgboost_10_score)
svm_linear_10_param = get_param(score,svm_linear_10_score )
svm_poly_10_param =  get_param(score,svm_poly_10_score)
svm_rbf_10_param =  get_param(score,svm_rbf_10_score)
rdforest_10_param =  get_param(score,rdforest_10_score )
knn_10_param=  get_param(score,knn_10_score)
nb_gauss_10_param =  get_param(score,nb_gauss_10_score)
nb_bernoulli_10_param =  get_param(score,nb_bernoulli_10_score)

elasticnet_20_param = get_param(score,elasticnet_20_score)
xgboost_20_param =  get_param(score,xgboost_20_score)
svm_linear_20_param = get_param(score,svm_linear_20_score )
svm_poly_20_param =  get_param(score,svm_poly_20_score)
svm_rbf_20_param =  get_param(score,svm_rbf_20_score)
rdforest_20_param =  get_param(score,rdforest_20_score )
knn_20_param=  get_param(score,knn_20_score)
nb_gauss_20_param =  get_param(score,nb_gauss_20_score)
nb_bernoulli_20_param =  get_param(score,nb_bernoulli_20_score)

elasticnet_10_reliefF_param = get_param(score,elasticnet_10_reliefF_score)
xgboost_10_reliefF_param =  get_param(score,xgboost_10_reliefF_score)
svm_linear_10_reliefF_param = get_param(score,svm_linear_10_reliefF_score )
svm_poly_10_reliefF_param =  get_param(score,svm_poly_10_reliefF_score)
svm_rbf_10_reliefF_param =  get_param(score,svm_rbf_10_reliefF_score)
rdforest_10_reliefF_param =  get_param(score,rdforest_10_reliefF_score )
knn_10_reliefF_param=  get_param(score,knn_10_reliefF_score)
nb_gauss_10_reliefF_param =  get_param(score,nb_gauss_10_reliefF_score)
nb_bernoulli_10_reliefF_param =  get_param(score,nb_bernoulli_10_reliefF_score)



elasticnet_20_reliefF_param = get_param(score,elasticnet_20_reliefF_score)
xgboost_20_reliefF_param =  get_param(score,xgboost_20_reliefF_score)
svm_linear_20_reliefF_param = get_param(score,svm_linear_20_reliefF_score )
svm_poly_20_reliefF_param =  get_param(score,svm_poly_20_reliefF_score)
svm_rbf_20_reliefF_param =  get_param(score,svm_rbf_20_reliefF_score)
rdforest_20_reliefF_param =  get_param(score,rdforest_20_reliefF_score )
knn_20_reliefF_param=  get_param(score,knn_20_reliefF_score)
nb_gauss_20_reliefF_param =  get_param(score,nb_gauss_20_reliefF_score)
nb_bernoulli_20_reliefF_param =  get_param(score,nb_bernoulli_20_reliefF_score)







elasticnet_full_param =  get_param(score,elasticnet_full_score)
xgboost_full_param =  get_param(score,xgboost_full_score)
svm_linear_full_param =  get_param(score,svm_linear_full_score)
svm_poly_full_param =  get_param(score,svm_poly_full_score)
svm_rbf_full_param =  get_param(score,svm_rbf_full_score)
rdforest_full_param =  get_param(score,rdforest_full_score)
knn_full_param=  get_param(score,knn_full_score)
nb_gauss_full_param =  get_param(score,nb_gauss_full_score)
nb_bernoulli_full_param =  get_param(score,nb_bernoulli_full_score )

elasticnet_sfs_param =  get_param(score,elasticnet_sfs_score)
xgboost_sfs_param =  get_param(score,xgboost_sfs_score)
svm_linear_sfs_param =  get_param(score,svm_linear_sfs_score)
svm_poly_sfs_param =  get_param(score,svm_poly_sfs_score)
svm_rbf_sfs_param =  get_param(score,svm_rbf_sfs_score)
rdforest_sfs_param =  get_param(score,rdforest_sfs_score)
knn_sfs_param=  get_param(score,knn_sfs_score)
nb_gauss_sfs_param =  get_param(score,nb_gauss_sfs_score)
nb_bernoulli_sfs_param =  get_param(score,nb_bernoulli_sfs_score )

elasticnet_CFS_param =  get_param(score,elasticnet_CFS_score)
xgboost_CFS_param =  get_param(score,xgboost_CFS_score)
svm_linear_CFS_param =  get_param(score,svm_linear_CFS_score)
svm_poly_CFS_param =  get_param(score,svm_poly_CFS_score)
svm_rbf_CFS_param =  get_param(score,svm_rbf_CFS_score)
rdforest_CFS_param =  get_param(score,rdforest_CFS_score)
knn_CFS_param=  get_param(score,knn_CFS_score)
nb_gauss_CFS_param =  get_param(score,nb_gauss_CFS_score)
nb_bernoulli_CFS_param =  get_param(score,nb_bernoulli_CFS_score )



                    ########### boosting ############
                    
###########                     #########                #########



svm_linear_dict_10_boost=[]
svm_poly_dict_10_boost=[]
svm_rbf_dict_10_boost=[]
rdforest_dict_10_boost=[]
knn_dict_10_boost=[]
nb_gauss_dict_10_boost =[]
nb_bernoulli_dict_10_boost = []




for i in range(10):
   
    dict = boost_knn(data,'HPYLORI', 10, 'infogain',knn_10_param)
    knn_dict_10_boost.append(dict)
    
    
    linear = boost_svm(data,'HPYLORI', 10, 'infogain',svm_linear_10_param,'linear')
    svm_linear_dict_10_boost.append(linear)
    
    poly = boost_svm(data,'HPYLORI', 10, 'infogain',svm_linear_10_param,'poly')
    svm_poly_dict_10_boost.append(poly)
    
    rbf = boost_svm(data,'HPYLORI', 10, 'infogain',svm_linear_10_param,'rbf')
    svm_rbf_dict_10_boost.append(rbf)
    
    
    dict = boost_rdforest(data,'HPYLORI', 10, 'infogain',rdforest_10_param)
    rdforest_dict_10_boost.append(dict)
 
    
    gauss= boost_naive_bayes(data,'HPYLORI', 10, 'infogain',nb_gauss_10_param,'Gaussian')
    nb_gauss_dict_10_boost.append(gauss)
    
    bernoulli= boost_naive_bayes(data,'HPYLORI', 10, 'infogain',nb_bernoulli_10_param,'Bernoulli')
    nb_bernoulli_dict_10_boost.append(bernoulli)
    

############## 20 features infogain ###########



svm_linear_dict_20_boost=[]
svm_poly_dict_20_boost=[]
svm_rbf_dict_20_boost=[]
rdforest_dict_20_boost=[]
knn_dict_20_boost=[]
nb_gauss_dict_20_boost =[]
nb_bernoulli_dict_20_boost = []




for i in range(10):
   
    dict = boost_knn(data,'HPYLORI', 20, 'infogain',knn_20_param)
    knn_dict_20_boost.append(dict)
    
    
    linear = boost_svm(data,'HPYLORI', 20, 'infogain',svm_linear_20_param,'linear')
    svm_linear_dict_20_boost.append(linear)
    
    poly = boost_svm(data,'HPYLORI', 20, 'infogain',svm_linear_20_param,'poly')
    svm_poly_dict_20_boost.append(poly)
    
    rbf = boost_svm(data,'HPYLORI', 20, 'infogain',svm_linear_20_param,'rbf')
    svm_rbf_dict_20_boost.append(rbf)
    
    
    dict = boost_rdforest(data,'HPYLORI', 20, 'infogain',rdforest_20_param)
    rdforest_dict_20_boost.append(dict)
 
    
    gauss= boost_naive_bayes(data,'HPYLORI', 20, 'infogain',nb_gauss_20_param,'Gaussian')
    nb_gauss_dict_10_boost.append(gauss)
    
    bernoulli= boost_naive_bayes(data,'HPYLORI', 20, 'infogain',nb_bernoulli_20_param,'Bernoulli')
    nb_bernoulli_dict_20_boost.append(bernoulli)


########### 10 features reliefF ##############

svm_linear_reliefF_10_boost=[]
svm_poly_reliefF_10_boost=[]
svm_rbf_reliefF_10_boost=[]
rdforest_reliefF_10_boost=[]
knn_reliefF_10_boost=[]
nb_gauss_reliefF_10_boost =[]
nb_bernoulli_reliefF_10_boost = []


for i in range(10):
   
    dict = boost_knn(data,'HPYLORI', 10, 'infogain',knn_10_reliefF_param)
    knn_reliefF_10_boost.append(dict)
    
    
    linear = boost_svm(data,'HPYLORI', 10, 'infogain',svm_linear_10_reliefF_param,'linear')
    svm_linear_reliefF_10_boost.append(linear)
    
    poly = boost_svm(data,'HPYLORI', 10, 'infogain',svm_poly_10_reliefF_param,'poly')
    svm_poly_reliefF_10_boost.append(poly)
    
    rbf = boost_svm(data,'HPYLORI', 10, 'infogain',svm_rbf_10_reliefF_param ,'rbf')
    svm_rbf_reliefF_10_boost.append(rbf)
    
    
    dict = boost_rdforest(data,'HPYLORI', 10, 'infogain',rdforest_10_reliefF_param)
    rdforest_reliefF_10_boost.append(dict)
 
    
    gauss= boost_naive_bayes(data,'HPYLORI', 10, 'infogain',nb_gauss_10_reliefF_param,'Gaussian')
    nb_gauss_reliefF_10_boost.append(gauss)
    
    bernoulli= boost_naive_bayes(data,'HPYLORI', 10, 'infogain',nb_bernoulli_10_reliefF_param,'Bernoulli')
    nb_bernoulli_reliefF_10_boost.append(bernoulli)


########## 20 features reliefF #############


svm_linear_reliefF_20_boost=[]
svm_poly_reliefF_20_boost=[]
svm_rbf_reliefF_20_boost=[]
rdforest_reliefF_20_boost=[]
knn_reliefF_20_boost=[]
nb_gauss_reliefF_20_boost =[]
nb_bernoulli_reliefF_20_boost = []


for i in range(10):
   
    dict = boost_knn(data,'HPYLORI', 20, 'infogain',knn_20_reliefF_param)
    knn_reliefF_20_boost.append(dict)
    
    
    linear = boost_svm(data,'HPYLORI', 20, 'infogain',svm_linear_20_reliefF_param,'linear')
    svm_linear_reliefF_20_boost.append(linear)
    
    poly = boost_svm(data,'HPYLORI', 20, 'infogain',svm_poly_20_reliefF_param,'poly')
    svm_poly_reliefF_20_boost.append(poly)
    
    rbf = boost_svm(data,'HPYLORI', 20, 'infogain',svm_rbf_20_reliefF_param ,'rbf')
    svm_rbf_reliefF_20_boost.append(rbf)
    
    
    dict = boost_rdforest(data,'HPYLORI', 20, 'infogain',rdforest_20_reliefF_param)
    rdforest_reliefF_20_boost.append(dict)
 
    
    gauss= boost_naive_bayes(data,'HPYLORI', 20, 'infogain',nb_gauss_20_reliefF_param,'Gaussian')
    nb_gauss_reliefF_20_boost.append(gauss)
    
    bernoulli= boost_naive_bayes(data,'HPYLORI', 20, 'infogain',nb_bernoulli_20_reliefF_param,'Bernoulli')
    nb_bernoulli_reliefF_20_boost.append(bernoulli)


############# full###############


svm_linear_full_boost=[]
svm_poly_full_boost=[]
svm_rbf_full_boost=[]
rdforest_full_boost=[]
knn_full_boost=[]
nb_gauss_full_boost =[]
nb_bernoulli_full_boost = []





for i in range(10):
    dict = boost_knn(data,'HPYLORI', data.shape[1]-1, 'infogain',knn_full_param)
    knn_full_boost.append(dict)
    
    
    linear = boost_svm(data,'HPYLORI', data.shape[1]-1, 'infogain',svm_linear_full_param,'linear')
    svm_linear_full_boost.append(linear)
    
    poly = boost_svm(data,'HPYLORI', data.shape[1]-1, 'infogain',svm_poly_full_param,'poly')
    svm_poly_full_boost.append(poly)
    
    rbf = boost_svm(data,'HPYLORI', data.shape[1]-1, 'infogain',svm_rbf_full_param ,'rbf')
    svm_rbf_full_boost.append(rbf)
    
    
    dict = boost_rdforest(data,'HPYLORI', data.shape[1]-1,, 'infogain',rdforest_full_param)
    rdforest_full_boost.append(dict)
 
    
    gauss= boost_naive_bayes(data,'HPYLORI', data.shape[1]-1,, 'infogain',nb_gauss_full_param,'Gaussian')
    nb_gauss_full_boost.append(gauss)
    
    bernoulli= boost_naive_bayes(data,'HPYLORI', data.shape[1]-1,, 'infogain',nb_bernoulli_full_param,'Bernoulli')
    nb_bernoulli_full_boost.append(bernoulli)

    
   
    
    
##################sfs##################

svm_linear_sfs_boost=[]
svm_poly_sfs_boost=[]
svm_rbf_sfs_boost=[]
rdforest_sfs_boost=[]
knn_sfs_boost=[]
nb_gauss_sfs_boost =[]
nb_bernoulli_sfs_boost = []





for i in range(10):
    dict = boost_knn(data,'HPYLORI', data.shape[1]-1, 'sfs',knn_sfs_param)
    knn_sfs_boost.append(dict)
    
    
    linear = boost_svm(data,'HPYLORI', data.shape[1]-1, 'sfs',svm_linear_sfs_param,'linear')
    svm_linear_sfs_boost.append(linear)
    
    poly = boost_svm(data,'HPYLORI', data.shape[1]-1, 'sfs',svm_poly_sfs_param,'poly')
    svm_poly_full_boost.append(poly)
    
    rbf = boost_svm(data,'HPYLORI', data.shape[1]-1, 'sfs',svm_rbf_sfs_param ,'rbf')
    svm_rbf_sfs_boost.append(rbf)
    
    
    dict = boost_rdforest(data,'HPYLORI', data.shape[1]-1,, 'sfs',rdforest_sfs_param)
    rdforest_sfs_boost.append(dict)
 
    
    gauss= boost_naive_bayes(data,'HPYLORI', data.shape[1]-1,, 'sfs',nb_gauss_sfs_param,'Gaussian')
    nb_gauss_sfs_boost.append(gauss)
    
    bernoulli= boost_naive_bayes(data,'HPYLORI', data.shape[1]-1,, 'sfs',nb_bernoulli_sfs_param,'Bernoulli')
    nb_bernoulli_sfs_boost.append(bernoulli)





    
    
    ################## CFS ##################


svm_linear_CFS_boost=[]
svm_poly_CFS_boost=[]
svm_rbf_CFS_boost=[]
rdforest_CFS_boost=[]
knn_CFS_boost=[]
nb_gauss_CFS_boost =[]
nb_bernoulli_CFS_boost = []





for i in range(10):
    dict = boost_knn(data,'HPYLORI', data.shape[1]-1, 'CFS',knn_CFS_param)
    knn_CFS_boost.append(dict)
    
    
    linear = boost_svm(data,'HPYLORI', data.shape[1]-1, 'CFS',svm_linear_CFS_param,'linear')
    svm_linear_sfs_boost.append(linear)
    
    poly = boost_svm(data,'HPYLORI', data.shape[1]-1, 'CFS',svm_poly_CFS_param,'poly')
    svm_poly_CFS_boost.append(poly)
    
    rbf = boost_svm(data,'HPYLORI', data.shape[1]-1, 'CFS',svm_rbf_CFS_param ,'rbf')
    svm_rbf_CFS_boost.append(rbf)
    
    
    dict = boost_rdforest(data,'HPYLORI', data.shape[1]-1,, 'CFS',rdforest_CFS_param)
    rdforest_CFS_boost.append(dict)
 
    
    gauss= boost_naive_bayes(data,'HPYLORI', data.shape[1]-1,, 'CFS',nb_gauss_CFS_param,'Gaussian')
    nb_gauss_CFS_boost.append(gauss)
    
    bernoulli= boost_naive_bayes(data,'HPYLORI', data.shape[1]-1,, 'CFS',nb_bernoulli_CFS_param,'Bernoulli')
    nb_bernoulli_CFS_boost.append(bernoulli)

























         











            

    
    
        

      

        
        
        
        

       
    
    




  









  
  
  
  
  
  
  
  
  
  
  

        
        
    
    















# losreg.fit(X=x_train, y=y_train)
# predictions = linreg.predict(X=x_test)
# error = predictions-y_test
# rmse = np.sqrt((np.sum(error**2)/len(x_test)))
# coefs = linreg.coef_
# features = x_train.columns
# '''


# '''
# #regularization
# alphas = np.linspace(0.0001, 1,100)
# rmse_list = []
# best_alpha = 0

# for a in alphas:
#     lasso = Lasso(fit_intercept = True, alpha = a, max_iter= 10000 )
    
#     kf = KFold(n_splits=10)
#     xval_err =0
    
    
    
    
#     for train_index, validation_index in kf.split(x_train):
    
#         lasso.fit(x_train.loc[train_index,:], y_train[train_index])
      
#         p = lasso.predict(x_train.iloc[validation_index,:])
#         err = p-y_train[validation_index]
#         xval_err = xval_err+np.dot(err,err)
#         rmse_10cv = np.sqrt(xval_err/len(x_train))
#         rmse_list.append(rmse_10cv)
#         best_alpha = alphas[rmse_list==min(rmse_list)]
      
        
# #using the alpha calculated to calculate accuracy of the test
# lasso = Lasso(fit_intercept = True, alpha = best_alpha)
# lasso.fit(x_train, y_train)
# predictionsOnTestdata_lasso = lasso.predict(x_test)
# predictionErrorOnTestData_lasso = predictionErrorOnTestData_lasso-y_test
# rmse_lasso 


        
        










    
