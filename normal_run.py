import data_preprocess as dp
import numpy as np
import pandas as pd
import featureselection as fselect
from sklearn.model_selection import StratifiedKFold
import classifiers as e

def normal_run(data, n_seed=5, splits=10, methods=['infogain_10'], estimators=['rdforest']):
    X = np.array(data.drop('HPYLORI',axis=1))
    y = np.array(data.HPYLORI)
    
    features = np.zeros(X.shape[1])
    outer_names = create_empty_dic(estimators, methods)

    for seed in range(n_seed):
        skf = StratifiedKFold(n_splits=splits, random_state=seed, shuffle=True)
        split_num = 0
        names = create_empty_dic(estimators, methods)
        
        for train, test in skf.split(X,y):
            split_num += 1
            X_train, X_test = X[train], X[test]
            y_train, y_test = y[train], y[test]
            X_train, X_test = dp.impute(X_train), dp.impute(X_test)
                    
            for s in methods:
                selector = s.split('_')[0]
                n_features = int(s.split('_')[1])
                selection = fselect.run_feature_selection(selector, X_train, y_train, n_features)
                for i in selection:
                    features[i] += 1
                        
                X_train, X_test = X_train[:,selection], X_test[:,selection]
                
                for estimator in estimators:
                    
                    if len(estimator.split('_')) == 1:
                        result = e.classify(estimator, X_train, X_test, y_train, y_test)
                        filename = estimator+s+'_'+str(split_num)+'_'+str(seed)
                        names[estimator][s].append(filename)
                        result.to_csv(filename)
                    elif estimator.split('_')[1]=='boost':
                        n_ests = [25,50]
                        rates = np.linspace(0.1,0.5,2)
                        for n_est in n_ests:
                            for rate in rates:
                                result = e.classify(estimator, X_train, X_test, y_train, y_test, n_est, rate)
                                filename = estimator+s+'_'+str(split_num)+'_'+str(seed)+'_'+str(n_est)+'_'+str(rate)+'.csv'
                                names[estimator][s].append(filename)
                                result.to_csv(filename)
                    elif estimator.split('_')[1]=='bag':
                        n_ests = [25,50]
                        for n_est in n_ests:
                            result = e.classify(estimator, X_train, X_test, y_train, y_test, n_est)
                            filename = estimator+s+'_'+str(split_num)+'_'+str(seed)+'_'+str(n_est)+'.csv'
                            names[estimator][s].append(filename)
                            result.to_csv(filename)
        
        create_interim_csv(names, outer_names, seed, splits)
        
    file_list = create_final_csv(outer_names, n_seed)
    
    final_features = pd.DataFrame([features],columns=data.drop('HPYLORI',axis=1).columns)
    final_features.to_csv('final_features.csv')

    return file_list




def subset_run(data, n_seed=5, splits=10, methods=['sfs'], estimators=['rdforest']):
    X = np.array(data.drop('HPYLORI',axis=1))
    y = np.array(data.HPYLORI)
    
    final_features = {}
    for estimator in estimators:
        final_features[estimator] = {}
        for method in methods:
            final_features[estimator][method] = np.zeros(X.shape[1])


    outer_names = create_empty_dic(estimators, methods)

    for seed in range(n_seed):
        skf = StratifiedKFold(n_splits=splits, random_state=seed, shuffle=True)
        split_num = 0
        names = create_empty_dic(estimators, methods)
        
        for train, test in skf.split(X,y):
            split_num += 1
            X_train, X_test = X[train], X[test]
            y_train, y_test = y[train], y[test]
            X_train, X_test = dp.impute(X_train), dp.impute(X_test)

            for estimator in estimators:
                for s in methods:
                    selector = s.split('_')[0]
                    n_features = int(s.split('_')[1])
                    selection = fselect.run_subset_selection(selector, X_train, y_train, n_features, estimator)
                    for i in selection:
                        final_features[estimator][method][i] += 1

                    X_train, X_test = X_train[:,selection], X_test[:,selection]

                    if len(estimator.split('_')) == 1:
                        result = e.classify(estimator, X_train, X_test, y_train, y_test)
                        filename = estimator+s+'_'+str(split_num)+'_'+str(seed)
                        names[estimator][s].append(filename)
                        result.to_csv(filename+'.csv')
                    elif estimator.split('_')[1]=='boost':
                        n_ests = [25,50]
                        rates = np.linspace(0.1,0.5,2)
                        for n_est in n_ests:
                            for rate in rates:
                                result = e.classify(estimator, X_train, X_test, y_train, y_test, n_est, rate)
                                filename = estimator+s+'_'+str(split_num)+'_'+str(seed)+'_'+str(n_est)+'_'+str(rate)
                                names[estimator][s].append(filename)
                                result.to_csv(filename+'.csv')
                    elif estimator.split('_')[1]=='bag':
                        n_ests = [25,50]
                        for n_est in n_ests:
                            result = e.classify(estimator, X_train, X_test, y_train, y_test, n_est)
                            filename = estimator+s+'_'+str(split_num)+'_'+str(seed)+'_'+str(n_est)
                            names[estimator][s].append(filename)
                            result.to_csv(filename+'.csv')

        create_interim_csv(names, outer_names, seed, splits)
        
    file_list = create_final_csv(outer_names, n_seed)
    
    final_features = pd.DataFrame([final_features])
    final_features.to_csv('final_features.csv')

    return file_list



def create_empty_dic(estimators, methods):
    final_dict = {}
    for estimator in estimators:
        final_dict[estimator] = {}
        for method in methods:
            final_dict[estimator][method] = []
    return final_dict

def stringconverter(string_vector):
    return np.fromstring(string_vector[1:-1], dtype=np.int, sep=',')

def create_interim_csv(inner_dict, outer_dict, seed, n_splits):
    for estimator in inner_dict:
        for method in inner_dict[estimator]:
            count = 0
            df = pd.DataFrame()
            for file in inner_dict[estimator][method]:
                if count == 0:
                    temp = pd.read_csv(str(file)+'.csv', index_col=0)
                    df = pd.concat([df,temp],axis=1)
                else:
                    temp = pd.read_csv(str(file)+'.csv', index_col=0)
                    df = pd.concat([df, temp.iloc[:,-1]],axis=1)
                count += 1
            filename = estimator+method+'_'+str(seed)+'_precombined'
            df.to_csv(filename+'.csv')
            outer_dict[estimator][method].append(filename)
            cmat = np.array(df.iloc[:,-n_splits:])
            with open(filename+'.npy', 'wb') as f:
                np.save(f, cmat)

def create_final_csv(outer_dict, n_seed):
    final_names = []
    vfunc = np.vectorize(stringconverter)

    for estimator in outer_dict:
        for method in outer_dict[estimator]:
            dimension = pd.read_csv(outer_dict[estimator][method][0]+'.csv').shape
            final_df = pd.read_csv(outer_dict[estimator][method][0]+'.csv',index_col=0).iloc[:,range(dimension[1]-6)]
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
            final_filename = estimator+method+'_final'
            final_names.append(final_filename)
            final_df.to_csv(final_filename+'.csv')
            cmat = np.array(final_df.iloc[:,-n_seed:])
            print(cmat)
            with open(final_filename+'.npy', 'wb') as d:
                np.save(d, cmat)
        
    return final_names
