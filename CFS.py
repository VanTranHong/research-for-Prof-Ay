import pandas as pd
import numpy as np
import math

def CFS(train_data,train_result,features):
    columns = train_data.columns
    dictionary = get_corr(train_data, train_result)
    dict = {}
    for col in columns:
        for sub in dictionary.keys():
            
            subset = sub+'_'+col
            score = evaluate(subset,dictionary)
            dict.update({subset:score})
            
       
        score = evaluate(col,dictionary)
        dict.update({col:score})
    print(dict)
    key_max = max(dict.keys(), key=(lambda k: dict[k]))
    return max_key
        
    


def evaluate(subset, dict):
    up = 0
    down = len(subset)
    for feature in subset:
        name = feature+'_target'
        up+=dict.get(name)
    
    for feature in subset:
        for another_feature in subset:
            if feature != another_feature:
                name = feature+'_'+another_feature
                down+=2*dict.get(name)
    return up/math.sqrt(down)
                
            
    
    


def get_corr(train_data, train_result):
    columns = train_data.columns
    dict = {}
    for col in columns:
        for another_col in columns:
            if col!= another_col:
                name = col+"_"+another_col
                correlation = corr(np.array(train_data[col]), np.array(train_data[another_col]))
                dict.update({name: correlation})
    for col in columns:
        name = col+'_target'
        correlation = corr(np.array(train_data[col]), np.array(train_result))
        dict.update({name: correlation})
    return dict
    
    
def corr(feature1, feature2):
    both = 0
    thesum = 0
    for i in range(len(feature1)):
        if feature1[i]==1 & feature2[i]==1:
            both+=1
        thesum+=feature1[i]
        thesum+=feature2[i]
    return 2*both/thesum
    
        
    
    