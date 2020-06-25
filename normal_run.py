import data_preprocess as dp
import numpy as np
import pandas as pd
import featureselection as fselect
from sklearn.model_selection import StratifiedKFold
import classifiers as e

def normal_run(data, n_seed=5, splits=10, methods=['infogain_10'], estimators=['rdforest']):
    vfunc = np.vectorize(stringconverter)

    X = np.array(data.drop('HPYLORI',axis=1))
    y = np.array(data.HPYLORI)
    
    features = np.zeros(X.shape[1])
    outer_names = {}

    for estimator in estimators:
        outer_names[estimator] = {}
        for method in methods:
            outer_names[estimator][method] = []

    for seed in range(n_seed):
        skf = StratifiedKFold(n_splits=splits, random_state=seed, shuffle=True)
        split_num = 0
        names = {}
        for estimator in estimators:
            names[estimator] = {}
            for method in methods:
                names[estimator][method] = []
        
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
                    filename = estimator+s+'_'+str(split_num)+'_'+str(seed)+'.csv'
                    names[estimator][s].append(filename)
                    result.to_csv(filename)
        
        
        for estimator in names:
            for method in names[estimator]:
                count = 0
                df = pd.DataFrame()
                for file in names[estimator][method]:
                    if count == 0:
                        temp = pd.read_csv(str(file), index_col=0)
                        df = pd.concat([df,temp],axis=1)
                    else:
                        temp = pd.read_csv(str(file), index_col=0)
                        df = pd.concat([df, temp.iloc[:,-1]],axis=1)
                    count += 1
                filename = estimator+method+'_'+str(seed)+'_precombined'
                df.to_csv(filename+'.csv')
                outer_names[estimator][method].append(filename)
                cmat = np.array(df.iloc[:,-5:])
                with open(filename+'.npy', 'wb') as f:
                    np.save(f, cmat)

    final_names = []

    for estimator in outer_names:
        for method in outer_names[estimator]:
            dimension = pd.read_csv(outer_names[estimator][method][0]+'.csv').shape
            final_df = pd.read_csv(outer_names[estimator][method][0]+'.csv',index_col=0).iloc[:,range(dimension[1]-6)]
            count = 0
            for file in outer_names[estimator][method]:
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
            cmat = np.array(final_df.iloc[:,len(df.columns)-5:])
            with open(final_filename+'.npy', 'wb') as d:
                np.save(d, cmat)
    
    final_features = pd.DataFrame([features],columns=data.drop('HPYLORI',axis=1).columns)
    final_features.to_csv('final_features.csv')

    return final_names

def stringconverter(string_vector):
    return np.fromstring(string_vector[1:-1], dtype=np.int, sep=',')