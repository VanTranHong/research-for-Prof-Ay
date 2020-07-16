import data_preprocess as dp
import numpy as np
import pandas as pd
import featureselection as fselect
from sklearn.model_selection import StratifiedKFold
import classifiers as e
import os

data_path = str(os.getcwd())+'/data/'
results_path = str(os.getcwd())+'/results/'

def normal_run(data, n_seed=2, splits=5, methods=['infogain_10'], estimators=['rdforest']):
    X = np.array(data.drop('HPYLORI',axis=1))
    y = np.array(data.HPYLORI)

    if not os.path.exists('data'):
        os.makedirs('data')
    if not os.path.exists('results'):
        os.makedirs('results')

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

                    result = e.classify(estimator, X_train, X_test, y_train, y_test)
                    filename = data_path+estimator+s+'_'+str(split_num)+'_'+str(seed)
                    names[estimator][s].append(filename)
                    result.to_csv(filename+'.csv')



        create_interim_csv(names, outer_names, seed, splits)
        delete_interim_csv(names)
    print(outer_names)
    file_list = create_final_csv(outer_names, n_seed)




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
            filename = data_path+estimator+method+'_'+str(seed)+'_precombined'
            df.to_csv(filename+'.csv')
            outer_dict[estimator][method].append(filename)
            cmat = np.array(df.iloc[:,-n_splits:])
            with open(filename+'.npy', 'wb') as f:
                np.save(f, cmat)

def delete_interim_csv(inner_dict):
    for estimator in inner_dict:
        for method in inner_dict[estimator]:
            for file in inner_dict[estimator][method]:
                if os.path.exists(str(file)+'.csv'):
                    os.remove(str(file)+'.csv')

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
                    print('non parallel')
                    print(a)
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
