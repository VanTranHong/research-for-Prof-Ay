import pandas as pd
import numpy as np
import math
from itertools import chain, combinations
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))  
def CFS(train_data,train_result,features):
    columns = train_data.columns
    # dictionary = get_corr(train_data, train_result)
    
    
    powersets = powerset(columns)
    dict = {}
    for subset in powersets:
        converted = list(subset)
        if len(converted)>0:
            name = ''
            for i in range(len(converted)):
                if i < len(converted)-1: 
                    name= name+converted[i]+'_'
                else:
                    name = name+converted[i]
            dict.update({name:0})
    
    for subset in dict.keys():
        score = evaluate(subset,dictionary)
        dict.update({subset:score})
        
   
    key_max = max(dict.keys(), key=(lambda k: dict[k]))
    return max_key



def evaluate(thesubset, dict):
    subset = thesubset.split('_')
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
    # powersets = powerset(columns)
    dict = {}
    # for subset in powersets:
    #     converted = list(subset)
    #     if len(converted)>0:
    #         name = ''
    #         for i in range(len(converted)):
    #             if i < len(converted)-1: 
    #                 name= name+converted[i]+'_'
    #             else:
    #                 name = name+converted[i]
    #         dict.update({name:0})
                    
        
    
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
        if feature1[i] and feature2[i]:
            both+=1
        thesum+=feature1[i]
        thesum+=feature2[i]
    return 2*both/thesum
    
        
    
    