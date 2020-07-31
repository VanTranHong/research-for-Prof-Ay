import data_preprocess as dp
import numpy as np
import pandas as pd
import featureselection as fselect
from sklearn.model_selection import StratifiedKFold
import classifiers as e
import os
from joblib import Parallel, delayed
import normal_run as nr
import bag
import boost
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


    
    
def boostbag_run(n_seed, splits, methods,estimators, runs,name, n_features):
    '''
    running boost or bag
    
    Args:
        n_seed(int): number of times we run splits-fold cross validation
        splits(int): number of folds in each cross validation
        methods(list): feature selection methods, can only be ranking method or non SFS subsetbased methods
        estimators(list): classifiers
        runs(list): data for splits
        name (string): boost or bag
        n_features(int): number of features

    Returns:
        List(list): final csv files that can be scored
    '''
    
    
    if not os.path.exists('dataparallel'):
        os.makedirs('dataparallel')
    if not os.path.exists('resultsparallel'):
        os.makedirs('resultsparallel')
    vfunc = np.vectorize(stringconverter)
    features = execute_feature_selection(runs, methods, n_features, n_seed, splits)
    methods = [method+name for method in methods]
    outer_dict = create_empty_dic(estimators, methods)
    if name == 'boost':
        Parallel(n_jobs=-1)(delayed(execute_a_boost_run)(i,runs[i],features[i], estimators, methods) for  i  in range(len(runs)))
    else:
        Parallel(n_jobs=-1)(delayed(execute_a_bag_run)(i,runs[i],features[i], estimators, methods) for  i  in range(len(runs)))

    combinefile(methods,estimators,splits,n_seed,len(runs), outer_dict)
    delete_interim_csv(estimators,methods,len(runs))
    file_list = create_final_csv(outer_dict, n_seed)
    return file_list






def execute_a_boost_run(index, run,features,estimators, methods):
    '''
    Executing a boost run for different combinations of estimators and  methods
    
    Args:
    index(int): index of the run
    run(list): of X_train, X_test, y_train, y_test
    features(list): of features of that runs for different methods of feature selections
    estimators(list): of classifiers
    methods(list): of feature selections
    
    Returns:
    
    
    '''
    
    for i in range(len(methods)):
        method = methods[i]
        selection = features[i]
        X_train, X_test = run[0], run[1]
        y_train, y_test = run[2], run[3]
        X_train, X_test = X_train[:,selection], X_test[:,selection]
      
        for estimator in estimators:
            result = boost.classify_boost(estimator, X_train, X_test, y_train, y_test)
            filename = data_path+estimator+method+'_'+str(index)
            result.to_csv(filename+'.csv')
            
            
def execute_a_bag_run(index, run,features,estimators, methods):
    '''
    Executing a bag run for different combinations of estimators and  methods
    
    Args:
    index(int): index of the run
    run(list): of X_train, X_test, y_train, y_test
    features(list): of features of that runs for different methods of feature selections
    estimators(list): of classifiers
    methods(list): of feature selections
    
    Returns:
    
    
    '''
    
    for i in range(len(methods)):
        method = methods[i]
        selection = features[i]
        X_train, X_test = run[0], run[1]
        y_train, y_test = run[2], run[3]
        X_train, X_test = X_train[:,selection], X_test[:,selection]
        for estimator in estimators:
            result = bag.classify_bag(estimator, X_train, X_test, y_train, y_test)
           
            filename = data_path+estimator+method+'_'+str(index)
            result.to_csv(filename+'.csv')
            
def create_empty_dic(estimators, methods):
    '''creating an empty dictionary whose keys are estimators. values is also a dictionay whose value is feature selection method
    Args:
    estimators(list): of classifiers
    methods(list): feature selection methods
    Returns:
    an empty dictionary
    
    '''
    
    
    final_dict = {}
    for estimator in estimators:
        final_dict[estimator] = {}
        for method in methods:
            final_dict[estimator][method] = []
    return final_dict
def stringconverter(string_vector):
    '''converting string to int'''

    return np.fromstring(string_vector[1:-1], dtype=np.int, sep=',')
def create_final_csv(outer_dict, n_seed):
    
    '''
    creating a final csv file by putting the confusion matrix of different cross validations of the same combination of feature selection and classifiers together
    Args:
    outer_dict: dictionary that contains the names of the files
    n_need: number of cross validations
    
    Returns: the names of the csv files that we create
    
    '''
    
    
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
def combinefile(methods,estimators,splits, n_seed,num_runs, outer_dict):
    ''' combining the results from all runs of the same seed each combination of estimator and feature selection methods into a bigger file and put that file into the dictionary
    this will sum the matrix for each split of the same seed and write into a csv file
    
    Args:
    
    methods(list): of feature selection methods
    estimators(list): of classifiers
    splits(int): numnber of folds in each cross validaton
    n_seed(int): number of cross validation
    outer_dict: dictonary where we will put the names of files created into
    
    Returns:
    None
    
    '''
    
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
def execute_feature_selection(runs, methods, n_features,n_seed,splits):
    '''
    executing feature selection on the runs in parallel
    Args:
        n_seed(int): number of times we run splits-fold cross validation
        splits(int): number of folds in each cross validation
        methods(list): feature selection methods, can only be ranking method or non SFS subsetbased methods
        runs(list): data for splits
        
        n_features(int): number of features

    Returns:
        List: of size number of runs (=n_seed * splits)
        each item in the list is a list of the same size as methods, each item in that list is the list of features selected for that method
    
    '''
    
    
    result = Parallel(n_jobs=-1)(delayed(execute_feature_selection_a_run)(run, methods) for  run in runs)
    # produce_features(result, n_features, methods, n_seed,splits)
    return result

def execute_feature_selection_a_run(run, methods):
    '''
    executing feature selections of a run for the methods
    
    Args:
        run(list): of  X_train, X_test, y_train, y_test
        methods (list): of feature selection methods

    Returns:
        List: A list of the same size as methods where each item is list of features selected for that method for that run
    
    '''
    
    
    X_train,  y_train = run[0], run[2]
    arr =[]
    for method in methods:
        selector = method.split('_')[0]
        n_features = int(method.split('_')[1])
        arr.append(fselect.run_feature_selection(selector, X_train, y_train, n_features))
    return arr


def produce_features(result, n_features,methods, n_seed, splits):
    '''
    producing features file for each seed run for each feature selection methods
    
    Args:
    result(list): result form feature selection methods
    n_features(int): the number of features in the dataset
    methods(list): of feature selection methods
    n_seed(int): number of splits-fold cross validation
    splits(int):number of folds in each cross validation
    
    Returns: None
    
    
    '''
    
    
    
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
            
def delete_interim_csv(estimators,methods,runs_len):
    '''deleting unimportant temporary csv files'''
    
    for estimator in estimators:
        for method in methods:
            for i in range(runs_len):
                file = estimator+method+'_'+str(i)
                if os.path.exists(file+'.csv'):
                    os.remove(file+'.csv')