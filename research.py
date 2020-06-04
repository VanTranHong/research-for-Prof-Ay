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
from algorithm import svm,rdforest,lasso, elasticNet,xgboost, naive_bayes
from data_preprocess import modify_data

from featureselection import infogain, reliefF, sfs,run_feature_selection




######this code is used to convert sav file to csv file when necessary######
# import pyreadstat as py
# from openpyxl.workbook import Workbook
# df, meta = py.read_sav('Merged Data_two School_Taye _AY_SP2020_ML_project.sav')
# writer=pd.ExcelWriter ("converted.xlsx")
# df.to_excel(writer, 'df')
# df.to_csv('csvfile.csv')
# writer.save()

##### THIS IS DONE FOR MATT######

##### this code convert excel to sav file when necessary####
# read_excel = pd.read_excel('Child Anemia Data Uncleaned.xlsx')
# read_excel.to_csv('Child Anemia not cleaned.csv', index = None, header = True)
# read_excel = pd.read_excel('Maternal Data Not Cleaned.xlsx')
# read_excel.to_csv('Marternal not cleaned data.csv', index = None, header = True)



# ##dropping rows with the same CASE ID done for Maternal
# motherdata = pd.read_csv('Marternal not cleaned data.csv',skipinitialspace=True, header = 0)

# dropduplicated = motherdata.drop_duplicates(subset = ['CASE ID'], keep = 'first')
# dropduplicated.to_excel("motheroutput.xlsx") 


# ##selecting the youngest child of each unique ID for Child
# childdata = pd.read_csv('Child Anemia not cleaned.csv',skipinitialspace=True, header = 0)
# print(childdata.shape)
# uniqueID = childdata['CASE ID'].unique()
# print(uniqueID.shape)
# returnframe = pd.DataFrame(columns=childdata.columns)

# for ID in uniqueID:
#     duplicates = childdata[childdata['CASE ID']==ID]
   
    
       
#     duplicates = duplicates.sort_values(by = 'Child Age in Months')
#     returnframe = returnframe.append(duplicates.iloc[0,:])
    
                           

# returnframe.to_excel("childoutput.xlsx") 



####handling the old data

#reading files and removing rows or columns with too many missing places
datafile = r'/Users/vantran/Documents/vantran/info.csv'


#######    main    ####
data = pd.read_csv(datafile,skipinitialspace=True, header = 0)

data.drop(['COLGATEID', 'Unnamed: 0'], axis = 1, inplace = True)

data = modify_data(data, [],data.columns)
writer = pd.ExcelWriter("imputed_data.xlsx")
    
data.to_excel(writer,'imputed data')
    # columns = [k for k in top_features]
    # values = [v for v in top_features.values()]
    # names_scores = list(zip( columns, values))
    # ns_df = pd.DataFrame(data = names_scores, columns=['feature', 'scores'])
    # #Sort the dataframe for better visualization
    # ns_df_sorted = ns_df.sort_values( by = ['scores'], ascending = False)
    # ns_df_sorted.to_excel(writer,'Features')
    
writer.save() 





####### Base case #########
# print('shape of data', data.shape)


#### feature selection ########
# top_features = infogain(data,'HPYLORI',10)

# top_features = reliefF(data,'HPYLORI',10 )
# top_features = sfs(data,'HPYLORI',10)

#########  algorithm    #########


def run_algorithms(features, method, file_to_write):
    svm(data,'HPYLORI', features, method, file_to_write)
    # rdforest(data,'HPYLORI',features, method, file_to_write)
    lasso(data,'HPYLORI', features, method, file_to_write)
    elasticNet(data,'HPYLORI', features, method, file_to_write)
    # xgboost(data,'HPYLORI',features, method, file_to_write)


    naive_bayes(data,'HPYLORI',features, method, file_to_write)
    
######## creating the excel files  
# wb = Workbook()
# wb.save('lasso.xlsx')

# # Load existing workbook
# wb = load_workbook('lasso.xlsx')
# # Designate sheet name and position

# sheet1 = wb.create_sheet('sheet1',0)
# lasso(data,'HPYLORI', features, method, sheet1)
# sheet2 = wb.create_sheet('sheet2',1)
# sheet3 = wb.create_sheet('sheet3',2)


f = open('research_result.txt','w')
rate = sum(data['HPYLORI'])/data.shape[0]
f.write('base line accuracy is '+str( max(rate,1-rate))+'\n')

feature_selection_methods = ['infogain','reliefF']
number_of_features = [10,15,20,25,data.shape[1]-1]



# rdforest(data,'HPYLORI', data.shape[1]-1, 'infogain', 'infogain_full_rdforest.xlsx')
# rdforest(data,'HPYLORI', 25, 'infogain', 'infogain_25_rdforest.xlsx')

# rdforest(data,'HPYLORI', data.shape[1]-1, 'reliefF', 'reliefF_full_rdforest.xlsx')
# rdforest(data,'HPYLORI', 15, 'reliefF', 'reliefF_15_rdforest.xlsx')

# naive_bayes(data,'HPYLORI', data.shape[1]-1, 'infogain', 'sfs_full_naive_bayes.xlsx')
# elasticNet(data,'HPYLORI', 15, 'sfs', 'sfs_15_elasticNet.xlsx')









# for method in feature_selection_methods:
#     for features in number_of_features:
#         f.write("feature selection method: "+ method+ " number of features used: "+ str(features))
#         run_algorithms(features, method, f)


f.close()



         











            

    
    
        

      

        
        
        
        

       
    
    




  









  
  
  
  
  
  
  
  
  
  
  

        
        
    
    















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


        
        










    
