import data_preprocess as dp
import numpy as np
import pandas as pd
import scoring as score
import normal_run as nr
import normal_run_parallel_ranked as parallel
import featureselection as fselect
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import  SGDClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import statsmodels.api as sm

#######    main    ####
def runSKFold(n_seed, splits, data):
    runs = []

    X = np.array(data.drop('Disease',axis=1))
    y = np.array(data.Disease)

    for seed in range(n_seed):
        skf = StratifiedKFold(n_splits=splits, random_state=seed, shuffle=True)
        for train, test in skf.split(X, y):
            X_train, X_test = X[train], X[test]
            X_train, X_test = dp.impute(X_train), dp.impute(X_test)
            y_train, y_test = y[train], y[test]
            arr = [X_train, X_test, y_train, y_test]
            runs.append(arr)
    return runs

def stats(data, column, file_to_write):
    logit = sm.Logit(data[column], data.drop(columns = [column]), axis =1)
    result = logit.fit()

    # fittedvalues = data[:, 2]
    # predict_mean_se  = data[:, 3]
    p_values = result.pvalues.to_frame()
    p_values.columns = [''] * len(p_values.columns)

    CI = result.conf_int(alpha = 0.05)
    CI = np.exp(CI)
    CI.columns = [''] * len(CI.columns)

    coeff = np.exp(result.params.to_frame())
    coeff.columns = [''] * len(coeff.columns)

    file_to_write.write('pvalues:')
    file_to_write.write(p_values.to_string()+'\n')

    file_to_write.write('CI 95%:')
    file_to_write.write(CI.to_string()+'\n')
    file_to_write.write('odds ratio:')
    file_to_write.write(coeff.to_string()+'\n')

def univariate_stats(data,column, file_to_write):
    file_to_write.write('univariate wise: \n')
    columns = data.drop(columns = [column]).columns
    for col in columns:
        logit = sm.Logit(data[column], data[col], axis =1)
        coeff = np.exp(logit.fit().params.to_frame())
        coeff.columns = [''] * len(coeff.columns)

        file_to_write.write(coeff.to_string())


def impute(data):
    columns = data.columns
    imputed_data = []
    for i in range(5):
        imputer = IterativeImputer(sample_posterior=True, random_state=i, verbose=1)
        imputed_data.append(imputer.fit_transform(data))
    returned_data = np.round(np.mean(imputed_data,axis = 0))
    return_data = pd.DataFrame(data = returned_data, columns=columns)




    return return_data














data = pd.read_csv('test.csv',skipinitialspace=True, header = 0)



data1 = dp.modify_data(data,[],data.columns)
print(data1)


############ producing univariate and multivariet regression #########

stats_file = open('Stats.txt','w')
imputed_data = impute(data1)

stats(imputed_data,'Disease',stats_file)
univariate_stats(imputed_data,'Disease',stats_file)
stats_file.close()


# ######## baseline accuracy ##########
f = open('research_result.txt','w')
rate = sum(imputed_data['Disease'])/imputed_data.shape[0]
rate2 = 1-rate
f.write('base line accuracy is '+str( max(rate,1-rate))+'\n')
f1 = 2*rate/(1+rate)
f.write('base line f1 value is '+str(f1))
f.close()





#################### RUNNING WITHOUT BOOSTING AND BAGGING for all ranking feature selections and CFS###############
runs = runSKFold(n_seed=5,splits=10,data=data1)


# # score.score(nr.normal_run(data1,n_seed=1,splits=2,methods=['cfs_0'],estimators=['knn']),1,2)
# num = data1.shape[1]-1
score.score(parallel.normal_run( 5, 10, ['infogain_20'], ['elasticnet','knn'], runs),5,10)
#, 'infogain_20','reliefF_10','reliefF_20','infogain_'+str(num),'cfs_0'
# # , 'svm','naive_bayes','knn','xgboost'
# # , 'infogain_20', 'reliefF_10','reliefF_20','infogain_'+str(num)
score.score(parallel.boostbag_run( 5, 10, ['introgain_10'], ['elasticnet'], runs,'bag'),5,10)
score.score(parallel.subset_run(5,10,['elasticnet'],['accuracy'],runs,),5,10)
# parallel.subset_run(1,2,['knn'],['f1'],runs)

# nr.normal_run(data1,n_seed=1,splits=2,methods=['infogain_10'],estimators=['rdforest'])
# 'rdforest','knn','xgboost','svm',






















































































































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
