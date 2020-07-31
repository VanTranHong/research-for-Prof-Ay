import data_preprocess as dp
import numpy as np
import pandas as pd
import scoring as score
import normal_run as nr
import ranking_subset_run as rsr
import sfs_run as sfs_r
import boost_bag_run as bbr
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

    X = np.array(data.drop('HPYLORI',axis=1))
    y = np.array(data.HPYLORI)

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
    endog = data[column]
    features = data.drop(columns=[column],axis=1).columns
    exog = sm.add_constant(data.loc[:,features])
    
    
    
    logit = sm.Logit(endog,exog)
    result = logit.fit()#maxiter=10000,skip_hessian=True,disp=0,method='bfgs'

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
    columns = [['ADD'],['AnyAllr'],['AnyPars3'],['CookArea'],['DEWORM'],['GCOW'],['HCIGR6A'],['GCAT_1','GCAT_2'],['GDOG_1.0','GDOG_2.0'],['GELEC_1.0','GELEC_2.0'],['GFLOOR6A_1.0','GFLOOR6A_2.0','GFLOOR6A_9.0'],['GWASTE_1.0','GWASTE_2.0','GWASTE_3.0'],['AgeGroups_1.0','AgeGroups_2.0'],['FamilyGrouped_1.0','FamilyGrouped_2.0'],['ToiletType_1.0','ToiletType_2.0'],['WaterSource_1.0','WaterSource_2.0']]
    
    for col in columns:
        endog = data[column]

        exog = sm.add_constant(data.loc[:,col])
    
    
    
        logit = sm.Logit(endog,exog)
        result = logit.fit()#maxiter=10000,skip_hessian=True,disp=0,method='bfgs'
        p_values = result.pvalues.to_frame()
        p_values.columns = [''] * len(p_values.columns)

        CI = result.conf_int(alpha = 0.05)
        CI = np.exp(CI)
        CI.columns = [''] * len(CI.columns)

        coeff = np.exp(result.params.to_frame())
        coeff.columns = [''] * len(coeff.columns)

        file_to_write.write('pvalues: \n')
        file_to_write.write(p_values.to_string()+'\n')

        file_to_write.write('CI 95%:\n')
        file_to_write.write(CI.to_string()+'\n')
        file_to_write.write('odds ratio: \n')
        file_to_write.write(coeff.to_string()+'\n')
        
        
def impute(data):
    columns = data.columns
    imputed_data = []
    for i in range(5):
        imputer = IterativeImputer(sample_posterior=True, random_state=i, verbose=1)
        imputed_data.append(imputer.fit_transform(data))
    returned_data = np.round(np.mean(imputed_data,axis = 0))
    return_data = pd.DataFrame(data = returned_data, columns=columns)

    return return_data



def gen_stats(data, target):
    data1 = impute(data)

    stats_file1 = open('multivariate.txt','w')

    stats(data1,target,stats_file1)
    stats_file1.close()
    
    stats_file2 = open('univariate.txt','w')
    univariate_stats(data1,target,stats_file2)
    stats_file2.close()


# # # ######## baseline accuracy ##########
    f = open('research_result.txt','w')
    rate = sum(data1[target])/data1.shape[0]
    rate2 = 1-rate
    f.write('base line accuracy is '+str( max(rate,1-rate))+'\n')
    f1 = 2*rate/(1+rate)
    f.write('base line f1 value is '+str(f1))
    f.close()