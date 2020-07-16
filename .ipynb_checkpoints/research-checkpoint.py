import data_preprocess as dp
import numpy as np
import pandas as pd
import scoring as score
import normal_run as nr
import normal_run_parallel_ranked as parall
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


#######    main    ####
data = pd.read_csv('info.csv',skipinitialspace=True, header = 0)

data1 = dp.modify_data(data,[],data.columns)

knn_classifiers = []
svm_classifiers = []
rdforest_classifiers = []
xgboost_classifiers =[]
naive_bayes_classifiers = []
neighbors = [1,3,5,7,9]
for n in neighbors:
    knn = KNeighborsClassifier(n_neighbors=n,n_jobs=-1)
    knn_classifiers.append(knn)
    
    
###### SVM #####
Cs = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]
gammas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1]
degrees = [2,3,4,5]
for c in Cs:
    linear = LinearSVC(C=c, random_state=0, max_iter=10000)
    svm_classifiers.append(linear)
    for gamma in gammas:
        rbf = SVC(kernel = 'rbf', C=c, gamma=gamma, random_state=0, max_iter=10000)
        svm_classifiers.append(rbf)

        for degree in degrees:
            poly = SVC(kernel='poly', C=c, gamma=gamma, degree=degree, random_state=0, max_iter=10000)
            svm_classifiers.append(poly)
            
            
######  RDROREST ######
estimators = [100, 200,300, 400, 500]
max_depths = [10,30,50,70]

for estimator in estimators:
    for max_d in max_depths:
        rdf = RandomForestClassifier(n_estimators=estimator, max_depth=max_d,  random_state=0, n_jobs=-1)
        rdforest_classifiers.append(rdf)
        
        
######### XGBOOST #######
rate = 0.05
#xg_alpha = 0
#xg_lambda = 1
max_depth = np.linspace(2, 10, 4, dtype=int)
n_estimators= np.linspace(50, 450, 4, dtype=int)
colsample_bytree= np.linspace(0.5, 1, 4)
subsample = np.linspace(0.5,1,4)    
min_child_weight = np.linspace(1, 19, 5, dtype=int)
scale_pos_weight=np.linspace(3, 10,3, dtype=int)

for depth in max_depth:
    for estimators in n_estimators:
        for col in colsample_bytree:
            for sub in subsample:
                for child_weight in min_child_weight:
                    for pos_weight in scale_pos_weight:
                        xgb = XGBClassifier(booster='gbtree',learning_rate=rate,max_depth=depth,
                                            min_child_weight=child_weight,colsample_bytree=col,
                                            scale_pos_weight=pos_weight,n_estimators=estimators, subsample=sub)
                        xgboost_classifiers.append(xgb)
                        
#######         NAIVE_BAYES #######
naive_bayes_classifiers.append(GaussianNB())
naive_bayes_classifiers.append(BernoulliNB())









############ producing univariate and multivariet regression #########

#stats_file = open('Stats.txt','w')
#imputed_data = impute(data)
# print(imputed_data)
#stats(imputed_data,'HPYLORI',stats_file)
#univariate_stats(imputed_data,'HPYLORI',stats_file)
#stats_file.close()


######### baseline accuracy ##########
#f = open('research_result.txt','w')
#rate = sum(data['HPYLORI'])/data.shape[0]
#f.write('base line accuracy is '+str( max(rate,1-rate))+'\n')
#f.close()








##################### RUNNING WITHOUT BOOSTING AND BAGGING for all ranking feature selections and CFS###############

#score.score(nr.normal_run(data1,n_seed=1,splits=2,estimators=['rdforest']),1,2)
score.score(parallel.normal_run(data1, n_seed=2, splits=5, methods=['infogain_10'], estimators=['rdforest']))
    


    
    
   

























         











            

    
    
        

      

        
        
        
        

       
    
    




  









  
  
  
  
  
  
  
  
  
  
  

        
        
    
    















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


        
        










    
