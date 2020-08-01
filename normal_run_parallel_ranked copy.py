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


def produce_features(result, n_features,methods):
    
    n_runs = 10
    for i in range(len(methods)):
        arr = [0 for i in range(n_features)]
        method = methods[i]
        for j in range(5):
            for k in range(10):
                feature = result[j*10+k][i]
                for f in feature:
                    arr[f]+=1
            df = pd.DataFrame(data=arr)
            df.to_csv(results_path+method+str(j)+'.csv')
                
                
        # for j in range(n_runs):
        #     feature = result[j][i]
        #     print(feature)


            

        
###########when we have selected the best subset of parameters for sfs #####
def produce_features_sfs(result, n_features,methods, n_classifier):
    n_runs = 50
    for i in range(len(methods)):
        
        method = methods[i]
        for j in range(5):
            arr = [0 for i in range(n_features)]
            for k in range(10):
                features = result[j*10+k][i]
                for feature in features:
                    for f in feature:
                        arr[f]+=1
            df = pd.DataFrame(data=arr)
            df.to_csv(results_path+method+str(j)+'.csv')
                        
                
     






def execute_feature_selection(runs, methods):
    result = Parallel(n_jobs=-1)(delayed(execute_feature_selection_a_run)(run, methods) for  run in runs)

    produce_features(result, 27, methods)
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

def normal_run( n_seed, splits, methods, estimators, runs):
    if not os.path.exists('dataparallel'):
        os.makedirs('dataparallel')
    if not os.path.exists('resultsparallel'):
        os.makedirs('resultsparallel')
    vfunc = np.vectorize(stringconverter)
    features = execute_feature_selection(runs, methods)
    outer_dict = create_empty_dic(estimators, methods)

    Parallel(n_jobs=-1)(delayed(execute_a_run)(i,runs[i],features[i], estimators, methods) for  i  in range(len(runs)))

    combinefile(methods,estimators,splits,n_seed,len(runs), outer_dict)
    delete_interim_csv(estimators,methods,len(runs))
    file_list = create_final_csv(outer_dict, n_seed)
    return file_list

def execute_feature_selection_sfs(runs,classifiers, metric):
    result =  Parallel(n_jobs=-1)(delayed(execute_featureselection_sfs_a_run)(run, classifiers, metric) for  run in runs)
    methods = [classifier+metric for classifier in classifiers]
    produce_features_sfs(result,27,methods, len(classifiers))
    return result
def execute_featureselection_sfs_a_run(run, classifiers, metric):##[metric][classifier]
    X_train,  y_train = run[0], run[2] #[metric][classifier][parameters]
    return_arr =[]
    for classifier in classifiers:
        est = get_classifiers(classifier)

        results = Parallel(n_jobs=-1)(delayed(fselect.sfs)(X_train, y_train, estimator, metric) for  estimator  in est)
        return_arr.append(results)
    return return_arr
def delete_interim_csv(estimators,methods,runs_len):
    for estimator in estimators:
        for method in methods:
            for i in range(runs_len):
                file = estimator+method+'_'+str(i)
                if os.path.exists(file+'.csv'):
                    os.remove(file+'.csv')
def subset_run(n_seed, splits, estimators, metrics,runs):
    if not os.path.exists('dataparallel'):
        os.makedirs('dataparallel')
    if not os.path.exists('resultsparallel'):
        os.makedirs('resultsparallel')

    vfunc = np.vectorize(stringconverter)
    for metric in metrics:
        features = execute_feature_selection_sfs(runs,estimators,metric)### a list of features corresponding to each run
    #     Parallel(n_jobs=-1)(delayed(execute_subset_run)(i,runs[i],features[i], estimators, metric) for  i  in range(len(runs)))
    # outer_dict = create_empty_dic(estimators, metrics)
    # combinefile(metrics,estimators,splits,n_seed,len(runs), outer_dict)
    # file_list = create_final_csv(outer_dict, n_seed)
    # delete_interim_csv(estimators,metrics,len(runs))

    # return file_list
    
    
def boostbag_run(n_seed, splits, methods,estimators, runs,name):
    if not os.path.exists('dataparallel'):
        os.makedirs('dataparallel')
    if not os.path.exists('resultsparallel'):
        os.makedirs('resultsparallel')
    vfunc = np.vectorize(stringconverter)
    features = execute_feature_selection(runs, methods)
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
    for i in range(len(methods)):
        method = methods[i]
        selection = features[i]
        X_train, X_test = run[0], run[1]
        y_train, y_test = run[2], run[3]
        X_train, X_test = X_train[:,selection], X_test[:,selection]
        print(selection)
        for estimator in estimators:
            result = e.classify_boost(estimator, X_train, X_test, y_train, y_test)
            filename = data_path+estimator+method+'_'+str(index)
            result.to_csv(filename+'.csv')
def execute_a_bag_run(index, run,features,estimators, methods):
    for i in range(len(methods)):
        method = methods[i]
        selection = features[i]
        X_train, X_test = run[0], run[1]
        y_train, y_test = run[2], run[3]
        X_train, X_test = X_train[:,selection], X_test[:,selection]
        for estimator in estimators:
            result = e.classify_bag(estimator, X_train, X_test, y_train, y_test)
            print(result)
            filename = data_path+estimator+method+'_'+str(index)
            result.to_csv(filename+'.csv')

def execute_subset_run(index, run,features,estimators, metric):
    X_train, X_test = run[0], run[1]
    y_train, y_test = run[2], run[3]
    for i in range(len(estimators)):
        estimator = estimators[i]
        feature = features[i]
        result = e.classify_sfs(estimator, X_train, X_test, y_train, y_test, feature)

        filename = data_path+estimator+metric+'_'+str(index)
        result.to_csv(filename+'.csv')
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
def get_classifiers(classifier):
    est = []
    if classifier == 'knn':
        neighbors = [20]
        for n in neighbors:
            knn = KNeighborsClassifier(n_neighbors=n,n_jobs=-1)
            est.append(knn)
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
    elif classifier=='rdforest':
        ######  RDROREST ######
        estimators = [ 300]
        max_depths = [5]

        for estimator in estimators:
            for max_d in max_depths:
                rdf = RandomForestClassifier(n_estimators=estimator, max_depth=max_d,  random_state=0, n_jobs=-1)
                est.append(rdf)
    elif classifier=='xgboost':
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

    elif classifier=='elasticnet':
        alphas= [0.001]#, 0.01, 0.1, 1, 10, 100]
        l1s = [0.2]#,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

        for alpha in alphas:
            for l1 in l1s:

                regr = SGDClassifier(loss = 'log',alpha= alpha,penalty = 'elasticnet',l1_ratio=l1,random_state=0)

                est.append(regr)

    else:
            #######         NAIVE_BAYES #######
        # est.append(GaussianNB())
        est.append(BernoulliNB())
    return est


