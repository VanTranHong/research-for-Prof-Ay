import data_preprocess as dp
import numpy as np
import pandas as pd
import scoring as score
import normal_run as nr

#######    main    ####
data = pd.read_csv('info.csv',skipinitialspace=True, header = 0)

data1 = dp.modify_data(data,[],data.columns)
data1 = data1.dropna()

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

score.score(nr.normal_run(data1,n_seed=1,splits=2,estimators=['rdforest']),1,2)
    


    
    
   

























         











            

    
    
        

      

        
        
        
        

       
    
    




  









  
  
  
  
  
  
  
  
  
  
  

        
        
    
    















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


        
        










    
