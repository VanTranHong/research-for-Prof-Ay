import data_preprocess as dp
import numpy as np
import pandas as pd
import featureselection as fselect
from sklearn.model_selection import StratifiedKFold
import sffs
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


###########when we have selected the best subset of parameters for sfs #####
def produce_features_sfs(result, n_features,methods, n_seed,splits ):#[run][classifier]
   
    # producing features file for each seed run for each feature selection methods
    # this only records the features that give the best accuracy or F1 score
    
    # Args:
    # result(list): result form feature selection methods
    # n_features(int): the number of features in the dataset
    # methods(list): of feature selection methods
    # n_seed(int): number of splits-fold cross validation
    # splits(int):number of folds in each cross validation
    
    # Returns: None
    
    
  
    
    for i in range(len(methods)):
        method = methods[i]
        for j in range(n_seed):
            arr = [0 for i in range(n_features)]
            for k in range(splits):
                features = result[j*splits+k][i]
                for feature in features:
                    for f in feature:
                        arr[f]+=1
            df = pd.DataFrame(data=arr)
            df.to_csv(results_path+method+str(j)+'.csv')
            

def execute_feature_selection_sfs(runs,classifiers, metric,n_features, n_seed, n_splits):
    
    
    # executing feature selection on the runs in parallel
    # Args:
    #     n_seed(int): number of times we run splits-fold cross validation
    #     splits(int): number of folds in each cross validation
    #     methods(list): feature selection methods, can only be ranking method or non SFS subsetbased methods
    #     runs(list): data for splits
        
    #     n_features(int): number of features

    # Returns:
    #     List: of size number of runs (=n_seed * splits)
    #     each item in the list is a list of the same size as methods, each item in that list is the list of features selected for that method
    
    
    
    result =  Parallel(n_jobs=-1)(delayed(execute_featureselection_sfs_a_run)(run, classifiers, metric) for  run in runs)#[run][classifiers]
 
    return result
def execute_featureselection_sfs_a_run(run, classifiers, metric):##[metric][classifier] a list of length n_classifier
    
    
    # executing SFFS for each classifier in the classifiers based on metric
    
    # Args:
    #     run(list): of  X_train, X_test, y_train, y_test
    #     classifiers(list): of classifiers
    #     metric(string): accuracy or F1 score

    # Returns:
    #     List: A list of the same size as methods where each item is list of features selected for that method for that run
    
    
    
    
    X_train,  y_train = run[0], run[2] #[metric][classifier][parameters]
    return_arr =[]
    for classifier in classifiers:
        est = get_classifiers(classifier)

        results = Parallel(n_jobs=-1)(delayed(fselect.sfs)(X_train, y_train, estimator, metric) for  estimator  in est)
        return_arr.append(results)
    print(return_arr)
    return return_arr


def execute_best_feature_a_run(run, classifiers, metric):
    # executing feature selection for a run for the classifiers using the best hyperparameters that will maximize the metric
    # run(list): of X_train, X_test, y_train, y_test
    # classifiers: list of classifiers
    # metric(string): accuracy or F1 score
    
    
    X_train,  y_train = run[0], run[2] #[metric][classifier][parameters]
    return_arr =[]
    for classifier in classifiers:
        est = get_best_classifier(classifier,metric)
        results = Parallel(n_jobs=-1)(delayed(fselect.sfs)(X_train, y_train, estimator, metric) for  estimator  in est)
        return_arr.append(results)
    return return_arr
        
    
def execute_best_feature_sfs(runs, classifiers, metric, n_features, n_seed, splits):
    # executing feature selection in parallel for the classifiers using the best hyperparameters that will maximize the metric
    # and write the features selected for each classifier for each cross validation into csv files
    # run(list): of X_train, X_test, y_train, y_test
    # classifiers: list of classifiers
    # n_features(int): the number of features in our dataset
    # n_seed: the number of cross validaton
    # splits: the number of folds in each cross validation
    # metric(string): accuracy or F1 score
    
    
    result = Parallel(n_jobs=-1)(delayed(execute_best_feature_a_run)(run,classifiers,metric) for run in runs)
    methods = [classifier+metric for classifier in classifiers]
    produce_features_sfs(result, n_features,methods,n_seed,splits)
    
    

def subset_features(n_seed, splits, estimators,metrics, runs, n_features):
    # calling this file to get the best features using SFFS method
    
    
    if not os.path.exists('dataparallel'):
        os.makedirs('dataparallel')
    if not os.path.exists('resultsparallel'):
        os.makedirs('resultsparallel')


    vfunc = np.vectorize(stringconverter)
    for metric in metrics:
        features = execute_best_feature_sfs(runs,estimators, metric, n_features, n_seed,splits)
    
    

def subset_run(n_seed, splits, estimators, metrics,runs, n_features):
    
    # running SFFS
    
    # Args:
    #     n_seed(int): number of times we run splits-fold cross validation
    #     splits(int): number of folds in each cross validation
    #     methods(list): feature selection methods, can only be ranking method or non SFS subsetbased methods
    #     estimators(list): classifiers
    #     runs(list): data for splits
    #     name (string): boost or bag
    #     n_features(int): number of features

    # Returns:
    #     List(list): final csv files that can be scored
    
    
    
    if not os.path.exists('dataparallel'):
        os.makedirs('dataparallel')
    if not os.path.exists('resultsparallel'):
        os.makedirs('resultsparallel')

    vfunc = np.vectorize(stringconverter)
    for metric in metrics:
        features = execute_feature_selection_sfs(runs,estimators,metric,n_features, n_seed, splits)### a list of features corresponding to each run
        Parallel(n_jobs=-1)(delayed(execute_subset_run)(i,runs[i],features[i], estimators, metric) for  i  in range(len(runs)))
    outer_dict = create_empty_dic(estimators, metrics)
    combinefile(metrics,estimators,splits,n_seed,len(runs), outer_dict)
    file_list = create_final_csv(outer_dict, n_seed)
    delete_interim_csv(estimators,metrics,len(runs))

    return file_list

def execute_subset_run(index, run,features,estimators, metric):
    
    # Executing a subset run for different estimators using that metric and features accordingly
    
    # Args:
    # index(int): index of the run
    # run(list): of X_train, X_test, y_train, y_test
    # features(list): of features of that runs for different methods of feature selections
    # estimators(list): of classifiers
    # metric(string): accuracy or F1 score
    
    # Returns:
    # None
    
    
    
    
    X_train, X_test = run[0], run[1]
    y_train, y_test = run[2], run[3]
    for i in range(len(estimators)):
        estimator = estimators[i]
        feature = features[i]
        result = sffs.classify_sfs(estimator, X_train, X_test, y_train, y_test, feature)

        filename = data_path+estimator+metric+'_'+str(index)
        result.to_csv(filename+'.csv')
        
        
def combinefile(methods,estimators,splits, n_seed,num_runs, outer_dict):
    # combining the results from all runs of the same seed each combination of estimator and feature selection methods into a bigger file and put that file into the dictionary
    # this will sum the matrix for each split of the same seed and write into a csv file
    
    # Args:
    
    # methods(list): of feature selection methods
    # estimators(list): of classifiers
    # splits(int): numnber of folds in each cross validaton
    # n_seed(int): number of cross validation
    # outer_dict: dictonary where we will put the names of files created into
    
    # Returns:
    # None
    
    
    
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
def delete_interim_csv(estimators,methods,runs_len):
    #    deleting unimportant temporary csv files
    for estimator in estimators:
        for method in methods:
            for i in range(runs_len):
                file = estimator+method+'_'+str(i)
                if os.path.exists(file+'.csv'):
                    os.remove(file+'.csv')
        
def create_empty_dic(estimators, methods):
    
    # creating an empty dictionary whose keys are estimators. values is also a dictionay whose value is feature selection method
    # Args:
    # estimators(list): of classifiers
    # methods(list): feature selection methods
    # Returns:
    # an empty dictionary
    

    final_dict = {}
    for estimator in estimators:
        final_dict[estimator] = {}
        for method in methods:
            final_dict[estimator][method] = []
    return final_dict
def stringconverter(string_vector):
    #    converting string to int
    return np.fromstring(string_vector[1:-1], dtype=np.int, sep=',')
        
def create_final_csv(outer_dict, n_seed):
    
    # creating a final csv file by putting the confusion matrix of different cross validations of the same combination of feature selection and classifiers together
    # Args:
    # outer_dict: dictionary that contains the names of the files
    # n_need: number of cross validations
    
    # Returns: the names of the csv files that we create
    
    
    
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
def get_classifiers(classifier):
    # getting the classifiers that will be used for feature selection methods of SFFS
    # Args:
    # classifier(string): name of classifier
    
    # Returns:
    # list of classifiers for that classifier
    
    est = []
    if classifier == 'knn':
        neighbors = [5,10,12,14,16,20]
        for n in neighbors:
            knn = KNeighborsClassifier(n_neighbors=n,n_jobs=-1)
            est.append(knn)
    elif classifier == 'svm':
           ###### SVM #####
        Cs = [1e-1, 1, 1e1, 1e2, 1e3]
        gammas = [1,1e1]
        degrees = [2,3]
        for c in Cs:
            linear = LinearSVC(C=c, random_state=0, max_iter=10000)
            est.append(linear)
            for gamma in gammas:
                rbf = SVC(kernel = 'rbf', C=c, gamma=gamma, random_state=0, max_iter=10000)
                est.append(rbf)

                for degree in degrees:
                    poly = SVC(kernel='poly', C=c, gamma=gamma, degree=degree, random_state=0, max_iter=10000)
                    est.append(poly)
    elif classifier=='rdforest':
        ######  RDROREST ######
        estimators = [ 300,500]
        max_depths = [5,10,15]

        for estimator in estimators:
            for max_d in max_depths:
                rdf = RandomForestClassifier(n_estimators=estimator, max_depth=max_d,  random_state=0, n_jobs=-1)
                est.append(rdf)
    elif classifier=='xgboost':
          ######### XGBOOST #######
        rate = 0.05
        #xg_alpha = 0
        #xg_lambda = 1
        max_depth = [1,2,3,4]#np.linspace(2, 10, 4, dtype=int)
        n_estimators= [50,100,150]#np.linspace(50, 450, 4, dtype=int)
        # colsample_bytree= np.linspace(0.5, 1, 4)
        # subsample = np.linspace(0.5,1,4)
        # min_child_weight = np.linspace(1, 19, 5, dtype=int)
        # scale_pos_weight=np.linspace(3, 10,3, dtype=int)

        for depth in max_depth:
            for estimators in n_estimators:
                # for col in colsample_bytree:
                #     for sub in subsample:
                #         for child_weight in min_child_weight:
                #             for pos_weight in scale_pos_weight:
                xgb = XGBClassifier(booster='gbtree',learning_rate=rate,max_depth=depth,
                                    n_estimators=estimators)
                est.append(xgb)

    elif classifier=='elasticnet':
        alphas= [0.001,0.0001, 0.01]#, 0.1, 1]#, 10, 100]
       

        for alpha in alphas:
           

            regr = SGDClassifier(loss = 'log',alpha= alpha,penalty = 'l1',random_state=0)

            est.append(regr)

    else:
            #######         NAIVE_BAYES #######
        est.append(GaussianNB())
        est.append(BernoulliNB())
    return est

def get_best_classifier(classifier,metric):
    # getting the classifiers for that classifier name that will maximize metric
    # Args:
    # classifier(string): name of the classifier
    # metric(string): accuracy or F1 score
    
    
    est = []
    if classifier == 'knn'  and metric == 'accuracy':
        neighbors = [20]
        for n in neighbors:
            knn = KNeighborsClassifier(n_neighbors=n,n_jobs=-1)
            est.append(knn)
    elif classifier == 'knn':
        neighbors = [20]
        for n in neighbors:
            knn = KNeighborsClassifier(n_neighbors=n,n_jobs=-1)
            est.append(knn)
          
    elif classifier == 'svm' and metric == 'accuracy':
           ###### SVM #####
        Cs = [1e-1]
        gammas = [1e1]
        # degrees = [2,3]
        for c in Cs:
            # linear = LinearSVC(C=c, random_state=0, max_iter=10000)
            # est.append(linear)
            for gamma in gammas:
                rbf = SVC(kernel = 'rbf', C=c, gamma=gamma, random_state=0, max_iter=10000)
                est.append(rbf)

            #     for degree in degrees:
            #         poly = SVC(kernel='poly', C=c, gamma=gamma, degree=degree, random_state=0, max_iter=10000)
            #         est.append(poly)
            
    elif classifier == 'svm':
           ###### SVM #####
        Cs = [1e-1]
        gammas = [1e1]
        # degrees = [2,3]
        for c in Cs:
            # linear = LinearSVC(C=c, random_state=0, max_iter=10000)
            # est.append(linear)
            for gamma in gammas:
                rbf = SVC(kernel = 'rbf', C=c, gamma=gamma, random_state=0, max_iter=10000)
                est.append(rbf)

            #     for degree in degrees:
            #         poly = SVC(kernel='poly', C=c, gamma=gamma, degree=degree, random_state=0, max_iter=10000)
            #         est.append(poly)  
    elif classifier=='rdforest' and metric == 'accuracy':
        ######  RDROREST ######
        estimators = [ 300]
        max_depths = [5]

        for estimator in estimators:
            for max_d in max_depths:
                rdf = RandomForestClassifier(n_estimators=estimator, max_depth=max_d,  random_state=0, n_jobs=-1)
                est.append(rdf)
    elif classifier == 'rdforest':
        ######  RDROREST ######
        estimators = [ 300]
        max_depths = [5]

        for estimator in estimators:
            for max_d in max_depths:
                rdf = RandomForestClassifier(n_estimators=estimator, max_depth=max_d,  random_state=0, n_jobs=-1)
                est.append(rdf)
    
    elif classifier=='xgboost' and metric == 'accuracy':
          ######### XGBOOST #######
        rate = 0.05
        #xg_alpha = 0
        #xg_lambda = 1
        max_depth = [2]#np.linspace(2, 10, 4, dtype=int)
        n_estimators= [150]#np.linspace(50, 450, 4, dtype=int)
        # colsample_bytree= np.linspace(0.5, 1, 4)
        # subsample = np.linspace(0.5,1,4)
        # min_child_weight = np.linspace(1, 19, 5, dtype=int)
        # scale_pos_weight=np.linspace(3, 10,3, dtype=int)

        for depth in max_depth:
            for estimators in n_estimators:
                # for col in colsample_bytree:
                #     for sub in subsample:
                #         for child_weight in min_child_weight:
                #             for pos_weight in scale_pos_weight:
                xgb = XGBClassifier(booster='gbtree',learning_rate=rate,max_depth=depth,
                                    n_estimators=estimators)
                est.append(xgb)
                
    elif classifier == 'xgboost':
          ######### XGBOOST #######
        rate = 0.05
        #xg_alpha = 0
        #xg_lambda = 1
        max_depth = [2]#np.linspace(2, 10, 4, dtype=int)
        n_estimators= [150]#np.linspace(50, 450, 4, dtype=int)
        # colsample_bytree= np.linspace(0.5, 1, 4)
        # subsample = np.linspace(0.5,1,4)
        # min_child_weight = np.linspace(1, 19, 5, dtype=int)
        # scale_pos_weight=np.linspace(3, 10,3, dtype=int)

        for depth in max_depth:
            for estimators in n_estimators:
                # for col in colsample_bytree:
                #     for sub in subsample:
                #         for child_weight in min_child_weight:
                #             for pos_weight in scale_pos_weight:
                xgb = XGBClassifier(booster='gbtree',learning_rate=rate,max_depth=depth,
                                    n_estimators=estimators)
                est.append(xgb)
        

    elif classifier=='elasticnet' and metric == 'accuracy':
        alphas= [0.001]#, 0.01, 0.1, 1, 10, 100]
        l1s = [0.2]#,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

        for alpha in alphas:
            for l1 in l1s:

                regr = SGDClassifier(loss = 'log',alpha= alpha,penalty = 'elasticnet',l1_ratio=l1,random_state=0)

                est.append(regr)
                
    elif classifier=='elasticnet':
        alphas= [0.001]#, 0.01, 0.1, 1, 10, 100]
        l1s = [0.2]#,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

        for alpha in alphas:
            for l1 in l1s:

                regr = SGDClassifier(loss = 'log',alpha= alpha,penalty = 'elasticnet',l1_ratio=l1,random_state=0)

                est.append(regr)

    elif classifier == 'naive_bayes':
            #######         NAIVE_BAYES #######
        # est.append(GaussianNB())
        est.append(BernoulliNB())
        
    return est


