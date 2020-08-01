import data_preprocess as dp
import numpy as np
import pandas as pd
import featureselection as fselect
from sklearn.model_selection import StratifiedKFold
import classifiers as e
import os
from joblib import Parallel, delayed
import normal_run as nr
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import  ElasticNet
from sklearn.linear_model import  SGDClassifier

data_path = str(os.getcwd())+'/dataparallel/'
results_path = str(os.getcwd())+'/resultsparallel/'


def produce_features(result, n_features,methods, n_seed, splits):
    for i in range(len(methods)):
        method = methods[i]
        for j in range(n_seed):
            arr = [0 for i in range(n_features)]
            for k in range(splits):
                feature = result[j*splits+k][i]
                for f in feature:
                    arr[f]+=1
            df = pd.DataFrame(data=arr)
            df.to_csv(results_path+method+str(j)+'.csv')
                
def execute_feature_selection(runs, methods, n_features,n_seed,splits):
    result = Parallel(n_jobs=-1)(delayed(execute_feature_selection_a_run)(run, methods) for  run in runs)
    produce_features(result, n_features, methods, n_seed,splits)
    return result

def execute_feature_selection_a_run(run, methods):
    X_train,  y_train = run[0], run[2]
    arr =[]
    for method in methods:
        selector = method.split('_')[0]
        n_features = int(method.split('_')[1])
        arr.append(fselect.run_feature_selection(selector, X_train, y_train, n_features))
    return arr
def execute_a_run(index, run,features,estimators, methods):
    for i in range(len(methods)):
        method = methods[i]
        selection = features[i]
        X_train, X_test = run[0], run[1]
        y_train, y_test = run[2], run[3]
        X_train, X_test = X_train[:,selection], X_test[:,selection]
        for estimator in estimators:
            result = e.classify(estimator, X_train, X_test, y_train, y_test)
            filename = data_path+estimator+method+'_'+str(index)
            result.to_csv(filename+'.csv')

def combinefile(methods,estimators,splits, n_seed,num_runs, outer_dict):
    for estimator in estimators:
        for method in methods:
            for seed in range(n_seed):
                df = pd.DataFrame()
                temp = pd.read_csv(str(data_path+estimator+method+'_'+str((seed)*splits)+'.csv'), index_col=0)
                df = pd.concat([df,temp],axis=1)
                for i  in range(1,splits):
                    temp = pd.read_csv(str(data_path+estimator+method+'_'+str((seed)*splits+i)+'.csv'), index_col=0)
                    df = pd.concat([df, temp.iloc[:,-1]],axis=1)

                filename = data_path+estimator+method+'_'+str(seed)+'_precombined'
                df.to_csv(filename+'.csv')
                outer_dict[estimator][method].append(filename)
                cmat = np.array(df.iloc[:,-splits:])
                with open(filename+'.npy', 'wb') as f:
                    np.save(f, cmat)

def normal_run( n_seed, splits, methods, estimators, runs,n_features):
    if not os.path.exists('dataparallel'):
        os.makedirs('dataparallel')
    if not os.path.exists('resultsparallel'):
        os.makedirs('resultsparallel')
    vfunc = np.vectorize(stringconverter)
    features = execute_feature_selection(runs, methods,n_features, n_seed, splits)
    outer_dict = create_empty_dic(estimators, methods)

    Parallel(n_jobs=-1)(delayed(execute_a_run)(i,runs[i],features[i], estimators, methods) for  i  in range(len(runs)))

    combinefile(methods,estimators,splits,n_seed,len(runs), outer_dict)
    delete_interim_csv(estimators,methods,len(runs))
    file_list = create_final_csv(outer_dict, n_seed)
    return file_list


def delete_interim_csv(estimators,methods,runs_len):
    for estimator in estimators:
        for method in methods:
            for i in range(runs_len):
                file = estimator+method+'_'+str(i)
                if os.path.exists(file+'.csv'):
                    os.remove(file+'.csv')




def create_empty_dic(estimators, methods):
    final_dict = {}
    for estimator in estimators:
        final_dict[estimator] = {}
        for method in methods:
            final_dict[estimator][method] = []
    return final_dict
def stringconverter(string_vector):

    return np.fromstring(string_vector[1:-1], dtype=np.int, sep=',')
def create_final_csv(outer_dict, n_seed):
    final_names = []
    vfunc = np.vectorize(stringconverter)

    for estimator in outer_dict:
        for method in outer_dict[estimator]:
            dimension = pd.read_csv(outer_dict[estimator][method][0]+'.csv').shape
            final_df = pd.read_csv(outer_dict[estimator][method][0]+'.csv',index_col=0).iloc[:,range(dimension[1]-n_seed)]
            count = 0
            for file in outer_dict[estimator][method]:
                with open(file+'.npy', 'rb') as f:
                    a = np.load(f, allow_pickle=True)

                collapsed_array = []
                for i in range(len(a)):

                    array = np.array([vfunc(xi) for xi in a[i]])

                    array = np.sum(array,axis=0)
                    collapsed_array.append([array])
                final_df = pd.concat([final_df,pd.DataFrame(collapsed_array, columns=['Confusion Matrix'])], axis=1)
            final_filename = results_path+estimator+method+'_final'
            final_names.append(final_filename)
            final_df.to_csv(final_filename+'.csv')
            cmat = np.array(final_df.iloc[:,-n_seed:])

            with open(final_filename+'.npy', 'wb') as d:
                np.save(d, cmat)

    return final_names
