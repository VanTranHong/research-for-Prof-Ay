   
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

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB

from featureselection import infogain, reliefF, sfs,run_feature_selection




############ boosting base classifier ###############

def boost (classifier):
    abc = AdaBoostClassifier(n_estimators = 20, base_estimator=classifier, learning_rate =1)
    return abc

########## bagging base classifier ################

def bag(classifier):
    abc = BaggingClassifier(base_estimator=classifier,n_estimators=10, random_state=0)
    return abc

   #______________________ALGORITHM______________________________ 
########## this is SVM##########
def svm(data, column, features, method, file_to_write):
    skf = StratifiedKFold(n_splits=10)
    X = np.array(data.drop(columns = [column], axis = 1))
    y = np.array(data[column])
    top_features = {}
    Cs = [0.1, 1, 10, 100, 1000]
    dict = {'poly':[0]*len(Cs),'rbf':[0]*len(Cs), 'poly_boosted':[0]*len(Cs), 'poly_bagged':[0]*len(Cs), 'rbf_boosted':[0]*len(Cs), 'rbf_bagged':[0]*len(Cs)}
    
    
    
    columns = data.drop(columns = [column], axis = 1).columns 
    for column in columns:
        top_features.update({column:0})
        
    for train, test in skf.split(X,y):
        # svm(data,X,y , train,test) 
        # parameters = {'kernel':('poly', 'rbf'), 'C':[0.1, 1, 10, 100, 1000]}
        # # grid = GridSearchCV(SVC(probability=True,kernel = 'poly' ), param_grid=parameters, refit = True, verbose = 3)
        
        train_data, test_data = X[train], X[test]
        
        train_result, test_result = y[train], y[test] 
        train_data = pd.DataFrame(data = train_data, columns=columns)
        test_data = pd.DataFrame(data = test_data, columns=columns)
        features_selected = run_feature_selection(method,train_data,train_result,features, 'svm')
        
        print(features_selected)
        for feature in features_selected:
            top_features[feature]+=1 
        train_data = np.array(train_data[features_selected])
        
        test_data = np.array(test_data[features_selected])
        
       
        
        
        
        for kernel in ['poly', 'rbf']:
            for method  in ['','_bagged','_boosted']:
                name = kernel+method
                scores = []
                if method=='':
                    for C in Cs  :
                        boosted = SVC(probability=True,kernel = kernel, C=C)  
                        boosted.fit(train_data, train_result)   
                        predicted_label = boosted.predict(test_data)
                        # print('boosted svm','kernel is', kernel, 'C is', C)
                        score = metrics.accuracy_score(test_result,predicted_label)/10
                        scores.append(score)
                elif method =='_boosted':
                    for C in Cs  :
                        boosted = boost(SVC(probability=True,kernel = kernel, C=C)) 
                        boosted.fit(train_data, train_result)   
                        predicted_label = boosted.predict(test_data)
                        # print('boosted svm','kernel is', kernel, 'C is', C)
                        score = metrics.accuracy_score(test_result,predicted_label)/10
                        scores.append(score)
                else:
                    for C in Cs  :
                        boosted = bag(SVC(probability=True,kernel = kernel, C=C))  
                        boosted.fit(train_data, train_result)   
                        predicted_label = boosted.predict(test_data)
                        # print('boosted svm','kernel is', kernel, 'C is', C)
                        score = metrics.accuracy_score(test_result,predicted_label)/10
                        scores.append(score)
                    
                
                dict[name] = [a + b for a, b in zip(scores, dict[name])]
    writer = pd.ExcelWriter(file_to_write)
    df = pd.DataFrame([dict],index = ['Accuracy'])
    df.to_excel(writer,'Accuracy')
    columns = [k for k in top_features]
    values = [v for v in top_features.values()]
    names_scores = list(zip( columns, values))
    ns_df = pd.DataFrame(data = names_scores, columns=['feature', 'scores'])
    #Sort the dataframe for better visualization
    ns_df_sorted = ns_df.sort_values( by = ['scores'], ascending = False)
    ns_df_sorted.to_excel(writer,'Features')
    
    writer.save() 
    
    

                
                
            
        
        
        
        
        
        
     








        
        
                
            
            
    # print (' for kernel poly, accuracy scores for C = ', Cs)
    # print ([x / 10 for x in dict['poly']])
    # print (' for kernel rbf, accuracy scores for C = ', Cs)
    # print ([x / 10 for x in dict['rbf']])
                
        # for kernel in ['poly', 'rbf']:
        #     for C in   [0.1, 1, 10, 100, 1000]:
        #         bagged = bag(SVC(probability=True,kernel = kernel, C=C))  
        #         bagged.fit(train_data, train_result)   
        #         predicted_label = bagged.predict(test_data)
        #         print('bagged svm','kernel is', kernel, 'C is', C)
        #         print(metrics.accuracy_score(test_result,predicted_label))
                
    # sum_accuracy =0```
    # sum_precision = 0
    # sum_recall = 0
    # sum_f1 = 0
    
           
        # print(accuracy_score(test_result, predicted_label))
    #     sum_precision = precision_score(test_result, predicted_label,  average="macro")
    #     sum_recall+= recall_score(test_result, predicted_label,  average="macro")
    #     sum_f1+= f1_score(test_result, predicted_label,  average="macro")
        
    # print('average accuracy')
    # print(sum_accuracy/10)
        
        
        
#########Random Forest Classification #################        
def rdforest(data, column, features, method, file_to_write):
    skf = StratifiedKFold(n_splits=10)
    X = np.array(data.drop(columns = [column], axis = 1))
    y = np.array(data[column])
    top_features = {}
    estimators = np.linspace(start = 20, stop = 200, num = 10)
    dict = {'rdforest':[0]*len(estimators),'rdforest_bagged':[0]*len(estimators), 'rdforest_boosted':[0]*len(estimators)}
    
    columns = data.drop(columns = [column], axis = 1).columns 
    for column in columns:
        top_features.update({column:0})
    for train, test in skf.split(X,y):
        train_data, test_data = X[train], X[test]
        train_result, test_result = y[train], y[test]  
        train_data = pd.DataFrame(data = train_data, columns=columns)
        test_data = pd.DataFrame(data = test_data, columns=columns)
        features_selected = run_feature_selection(method,train_data,train_result,features, 'rdforest')
        for feature in features_selected:
            top_features[feature]+=1
        train_data = np.array(train_data[features_selected])
        
        test_data = np.array(test_data[features_selected])
       
        n_estimators = [int(x) for x in estimators]
        methods = ['','_boosted','_bagged']
        for method in methods:
            name = 'rdforest'+method
            scores = []
            if method=='':
                for n_est in n_estimators:
                    classifier = RandomForestClassifier(n_estimators=35,  random_state=0)
                    boosted = classifier
                    boosted.fit(train_data, train_result)   
                    predicted_label = boosted.predict(test_data)
                    score = metrics.accuracy_score(test_result,predicted_label)/10
                    
                    scores.append(score)
            elif method =='_boosted':
                for n_est in n_estimators: 
                    classifier = RandomForestClassifier(n_estimators=35,  random_state=0)
                    boosted = boost(classifier)
                    boosted.fit(train_data, train_result)   
                    predicted_label = boosted.predict(test_data)
                    score = metrics.accuracy_score(test_result,predicted_label)/10
                    scores.append(score)
            else:
                for n_est in n_estimators:
                    classifier = RandomForestClassifier(n_estimators=35, random_state=0)
                    bagged = bag(classifier)
                    bagged.fit(train_data, train_result)   
                    predicted_label = boosted.predict(test_data)
                    score = metrics.accuracy_score(test_result,predicted_label)/10
                    scores.append(score)
            
            dict[name] = [a + b for a, b in zip(scores, dict[name])]
            # print(dict[name])
    writer = pd.ExcelWriter(file_to_write)
    df = pd.DataFrame([dict],index = ['Accuracy'])
    df.to_excel(writer,'Accuracy')
    columns = [k for k in top_features]
    values = [v for v in top_features.values()]
    names_scores = list(zip( columns, values))
    ns_df = pd.DataFrame(data = names_scores, columns=['feature', 'scores'])
    #Sort the dataframe for better visualization
    ns_df_sorted = ns_df.sort_values( by = ['scores'], ascending = False)
    ns_df_sorted.to_excel(writer,'Features')
    
    writer.save()  
    
    
                    
                
                
        
    
            
            
            
            
            
                
            
        
    
  #####################Lasso Logictic Regression CV ################    
def lasso(data,column, features, method, file_to_write):
    skf = StratifiedKFold(n_splits=10)
    X = np.array(data.drop(columns = [column], axis = 1))
    y = np.array(data[column])
    alphas = np.logspace(-4, -0.5, 30)
    dict = {}
    top_features = {}
    for alpha in alphas:
        
        dict.update({alpha:0})
    columns = data.drop(columns = [column], axis = 1).columns 
    for column in columns:
        top_features.update({column:0})
    
   
    for train, test in skf.split(X,y):
        train_data, test_data = X[train], X[test]
        train_result, test_result = y[train], y[test] 
        
        
        train_data = pd.DataFrame(data = train_data, columns=columns)
        test_data = pd.DataFrame(data = test_data, columns=columns)
        features_selected = run_feature_selection(method,train_data,train_result,features, 'lasso')
        
        
        
        for feature in features_selected:
            top_features[feature]+=1
        train_data = np.array(train_data[features_selected])
        
        test_data = np.array(test_data[features_selected])
        for alpha in alphas:
            lasso = LogisticRegression()
             
            
            lasso.fit(train_data, train_result)   
            predicted_label = lasso.predict(test_data)>0.5
            
            dict[alpha] = dict[alpha]+metrics.accuracy_score(test_result,predicted_label)/10
            
    writer = pd.ExcelWriter(file_to_write)
    df = pd.DataFrame([dict],index = ['Accuracy'])
    df.to_excel(writer,'Accuracy')
    columns = [k for k in top_features]
    values = [v for v in top_features.values()]
    names_scores = list(zip( columns, values))
    ns_df = pd.DataFrame(data = names_scores, columns=['feature', 'scores'])
    #Sort the dataframe for better visualization
    ns_df_sorted = ns_df.sort_values( by = ['scores'], ascending = False)
    ns_df_sorted.to_excel(writer,'Features')
    
    writer.save()    
        
        
        
# ###########  ElasticNetCV  ##########      
        
def elasticNet (data, column, features, method, file_to_write):
    skf = StratifiedKFold(n_splits=10)
    X = np.array(data.drop(columns = [column], axis = 1))
    y = np.array(data[column])
    alphas= [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
    l1s = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    dict = {}
    top_features = {}
    for alpha in alphas:
        for l1 in l1s:
            name = str(alpha)+','+str(l1)
            dict.update({name:0})
    columns = data.drop(columns = [column], axis = 1).columns 
    for column in columns:
        top_features.update({column:0})
   
    
    
    
    for train, test in skf.split(X,y):
        
        train_data, test_data = X[train], X[test]
        train_result, test_result = y[train], y[test]  
        train_data = pd.DataFrame(data = train_data, columns=columns)
        test_data = pd.DataFrame(data = test_data, columns=columns)
        features_selected = run_feature_selection(method,train_data,train_result,features, 'elasticNet')
        
        for feature in features_selected:
            top_features[feature]+=1
        
        for alpha in alphas:
            for l1 in l1s:
                regr = ElasticNet(alpha = alpha,l1_ratio = l1,random_state=0)
        
                model = regr.fit(train_data, train_result)
                predicted_label = model.predict(test_data)>0.5
                name = str(alpha)+','+str(l1)
                dict[name] = dict[name]+ metrics.accuracy_score(test_result,predicted_label)/10
    
    writer = pd.ExcelWriter(file_to_write)
    df = pd.DataFrame([dict],index = ['Accuracy'])
    df.to_excel(writer,'Accuracy')
    columns = [k for k in top_features]
    values = [v for v in top_features.values()]
    names_scores = list(zip( columns, values))
    ns_df = pd.DataFrame(data = names_scores, columns=['feature', 'scores'])
    #Sort the dataframe for better visualization
    ns_df_sorted = ns_df.sort_values( by = ['scores'], ascending = False)
    ns_df_sorted.to_excel(writer,'Features')
    
    writer.save() 
    
    
                    

       

        
                            
       
                
                
    
    
    
    
    
# ########### decision tree classifier ##################
def xgboost(data, column, features, method, file_to_write):   
    skf = StratifiedKFold(n_splits=10) 
    X = np.array(data.drop(columns = [column], axis = 1))
    y = np.array(data[column])
    top_features = {}
    dict = {'score':0} 
    columns = data.drop(columns = [column],axis = 1).columns
    for column in columns:
        top_features.update({column:0})

    
        
    for train, test in skf.split(X,y):   
        train_data, test_data = X[train], X[test]
        train_result, test_result = y[train], y[test] 
        train_data = pd.DataFrame(data = train_data, columns=columns)
        test_data = pd.DataFrame(data = test_data, columns=columns) 
        features_selected = run_feature_selection(method,train_data,train_result,features, 'xgboost')
        
        for feature in features_selected:
            top_features[feature]+=1
            
        
        train_data = np.array(train_data[features_selected])
        
        test_data = np.array(test_data[features_selected])
        
        scoring = ['f1', 'recall', 'precision']

        # XGBoost space
        params_xgb = {'learning_rate': np.linspace(0, 1, 11),
                    'min_split_loss': np.linspace(0, 1, 6),
                    'max_depth': np.linspace(2, 10, 5, dtype=int),
                    'min_child_weight': np.linspace(1, 20, 20, dtype=int),
                    'colsample_bytree': np.linspace(0.5, 1, 6),
                    'reg_alpha': np.linspace(0, 1, 11),
                    'scale_pos_weight': np.linspace(4, 50, 24, dtype=int),
                    'n_estimators': np.linspace(10, 450, 12, dtype=int),
                    'reg_lambda': np.linspace(0, 10, 11),
                    }

        xgb= RandomizedSearchCV(estimator=XGBClassifier(booster='gbtree',
                                                                            objective='binary:logistic',
                                                                            
                                                                            nthread=1),
                                                    param_distributions=params_xgb,
                                                    n_iter=250,
                                                    scoring=scoring,
                                                
                                                    n_jobs=-1,
                                                   
                                                    verbose=1,
                                                    refit='f1')

        xgb.fit(train_data, train_result)
        ypred = xgb.predict(test_data)
        dict['score']+=metrics.accuracy_score(test_result,ypred)/10
    writer = pd.ExcelWriter(file_to_write)
    df = pd.DataFrame([dict],index = ['Accuracy'])
    df.to_excel(writer,'Accuracy')
    columns = [k for k in top_features]
    values = [v for v in top_features.values()]
    names_scores = list(zip( columns, values))
    ns_df = pd.DataFrame(data = names_scores, columns=['feature', 'scores'])
    #Sort the dataframe for better visualization
    ns_df_sorted = ns_df.sort_values( by = ['scores'], ascending = False)
    ns_df_sorted.to_excel(writer,'Features')
    
    writer.save() 
        
    
        

################ naive bayes classifier###############
def naive_bayes(data, column, features, method, file_to_write):
    skf = StratifiedKFold(n_splits=10) 
    X = np.array(data.drop(columns = [column], axis = 1))
    y = np.array(data[column])
    dict = {'nb_G':0,'nb_G_boost':0, 'nb_G_bag':0,'nb_B':0,'nb_B_boost':0, 'nb_B_bag':0,}
    top_features = {}
    
    
    columns = data.drop(columns = [column], axis = 1).columns 
    for column in columns:
        top_features.update({column:0})
        
    
    for train, test in skf.split(X,y):
        
        train_data, test_data = X[train], X[test]
        train_result, test_result = y[train], y[test]  
        train_data = pd.DataFrame(data = train_data, columns=columns)
        test_data = pd.DataFrame(data = test_data, columns=columns)
        features_selected = run_feature_selection(method,train_data,train_result,features, 'naive_bayes')
       
        
        for feature in features_selected:
            top_features[feature]+=1
            
        
        train_data = np.array(train_data[features_selected])
        
        test_data = np.array(test_data[features_selected])
        
        
        
        gnb = GaussianNB()
        predicted_label = gnb.fit(train_data, train_result).predict(test_data)
        dict['nb_G'] = dict['nb_G']+metrics.accuracy_score(test_result,predicted_label)/10
        gnb = boost(GaussianNB())
        predicted_label = gnb.fit(train_data, train_result).predict(test_data)
        dict['nb_G_boost'] = dict['nb_G_boost']+metrics.accuracy_score(test_result,predicted_label)/10
        gnb = bag(GaussianNB())
        predicted_label = gnb.fit(train_data, train_result).predict(test_data)
        dict['nb_G_bag'] = dict['nb_G_bag']+metrics.accuracy_score(test_result,predicted_label)/10
        
        gnb = BernoulliNB(binarize=0.0)
        predicted_label = gnb.fit(train_data, train_result).predict(test_data)
        dict['nb_B'] = dict['nb_B']+metrics.accuracy_score(test_result,predicted_label)/10
        gnb = boost(BernoulliNB(binarize=0.0))
        predicted_label = gnb.fit(train_data, train_result).predict(test_data)
        dict['nb_B_boost'] = dict['nb_B_boost']+metrics.accuracy_score(test_result,predicted_label)/10
        gnb = bag(BernoulliNB(binarize=0.0))
        predicted_label = gnb.fit(train_data, train_result).predict(test_data)
        dict['nb_B_bag'] = dict['nb_B_bag']+metrics.accuracy_score(test_result,predicted_label)/10
        
       
    writer = pd.ExcelWriter(file_to_write)
    df = pd.DataFrame([dict],index = ['Accuracy'])
    df.to_excel(writer,'Accuracy')
    columns = [k for k in top_features]
    values = [v for v in top_features.values()]
    names_scores = list(zip( columns, values))
    ns_df = pd.DataFrame(data = names_scores, columns=['feature', 'scores'])
    #Sort the dataframe for better visualization
    ns_df_sorted = ns_df.sort_values( by = ['scores'], ascending = False)
    ns_df_sorted.to_excel(writer,'Features')
    
    writer.save() 
        

    
    

                
                
            
        
        
        
        
        
        
     






