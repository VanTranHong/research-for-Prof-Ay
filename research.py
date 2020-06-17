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

features =[]

for i in range(10):
    dict,top_features = elasticNet(data,'HPYLORI', 10, 'infogain')
    elasticnet_dict_10.append(dict)
    features.append(top_features)
    
    dict, top_features = xgboost(data,'HPYLORI', 10, 'infogain')
    xgboost_dict_10.append(dict)
    features.append(top_features)
    
    dict, top_features = KNN(data,'HPYLORI', 10, 'infogain')
    knn_dict_10.append(dict)
    features.append(top_features)
    
    linear, poly, rbf, top_features = svm(data,'HPYLORI', 10, 'infogain')
    svm_linear_dict_10.append(linear)
    svm_poly_dict_10.append(poly)
    svm_rbf_dict_10.append(rbf)
    features.append(top_features)
    
    dict,top_features = rdforest(data,'HPYLORI', 10, 'infogain')
    rdforest_dict_10.append(dict)
    features.append(top_features)
    
    gauss, bernoulli, top_features = naive_bayes(data,'HPYLORI', 10, 'infogain')
    nb_gauss_dict_10.append(gauss)
    nb_bernoulli_dict_10.append(bernoulli)
    features.append(top_features)




    
    






############## 20 features infogain ###########


########### 10 features reliefF ##############


########## 20 features reliefF #############




############# full###############














# dictionary, top_features = xgboost(data,'HPYLORI', 10, 'infogain', 'infogain_naive_bayes.xlsx','f1')




# naive_bayes(data, 'HPYLORI',10, 'infogain','infogain_10_naive_bayes.xlsx')



    





















         











            

    
    
        

      

        
        
        
        

       
    
    




  









  
  
  
  
  
  
  
  
  
  
  

        
        
    
    















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


        
        










    
