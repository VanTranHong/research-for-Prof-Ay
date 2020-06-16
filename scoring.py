
# tn,fp,fn,tp
def get_f1(contingency):
    return 2*get_precision(contingency)*get_PPV(contingency)/(get_precision(contingency)+get_PPV(contingency))

def get_precision(contingency):
    return contingency[3]/(contingency[3]+contingency[1])
    
def get_PPV(contingency): # recall
    return contingency[3]/(contingency[3]+contingency[2])
    
def get_accuracy(contingency): 
    return (contingency[0]+contingency[3])/(contingency[0]+contingency[1]+contingency[2]+contingency[3])
        
def f1(dict, classfier):
    keys = dict.keys()
    f1_dict= {}
    for key in keys:
        contingency = dict.get(key)
        f1 = get_f1(contingency)
        f1_dict.update({key:f1})
    Keymax = max(f1_dict, key=f1_dict.get) 

    return max(f1_dict.values()), Keymax, dict.get(Keymax)
        
    
    
    
    
    
def accuracy(dict):
    keys = dict.keys()
    accuracy_dict= {}
    for key in keys:
        contingency = dict.get(key)
        accuracy = get_accuracy(contingency)
        accuracy_dict.update({key:accuracy})
    Keymax = max(accuracy_dict, key=accuracy_dict.get) 

    return max(accuracy_dict.values()), Keymax, dict.get(Keymax)
    
    
def precision(dict):
    
    keys = dict.keys()
    precision_dict= {}
    for key in keys:
        contingency = dict.get(key)
        precision = get_precision(contingency)
        precision_dict.update({key:precision})
    Keymax = max(precision_dict, key=precision_dict.get) 

    return max(precision_dict.values()), Keymax, dict.get(Keymax)
    
    
def PPV(dict):
    keys = dict.keys()
    PPV_dict= {}
    for key in keys:
        contingency = dict.get(key)
        PPV = get_PPV(contingency)
        PPV_dict.update({key:PPV})
    Keymax = max(PPV_dict, key=PPV_dict.get) 

    return max(PPV_dict.values()), Keymax, dict.get(Keymax)
    