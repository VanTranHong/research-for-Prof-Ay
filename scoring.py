import pandas as pd
import numpy as np

def accuracy(tp, tn, total):
    return (tp+tn)/total

def precision(tp, fp):
    if tp+fp==0:
        return 0

    return (tp/(tp+fp))

def recall(tp, fn):
    if tp+fn ==0:
        return 0

    return (tp/(tp+fn))

def f1(x, y):
    if x+y ==0:
        return 0
    return 2*((x*y)/(x+y))

def score(files, n_seed, n_folds):
    for filename in files:
        df = pd.read_csv(filename+'.csv', index_col=0)
        final_df = pd.DataFrame()

        with open(filename+'.npy', 'rb') as f:
            confusion_array = np.load(f, allow_pickle=True)
        print(confusion_array)
        average_array = np.zeros((len(confusion_array),4))

        for i in range(len(confusion_array)):
            acc = []
            prec = []
            rec = []
            f = []
            for j in range(n_seed):
                tn = confusion_array[i][j][0]
                fp = confusion_array[i][j][1]
                fn = confusion_array[i][j][2]
                tp = confusion_array[i][j][3]
                acc.append(accuracy(tp, tn, sum([tn,fp,fn,tp])))
                p = precision(tp, fp)
                prec.append(p)
                r = recall(tp, fn)
                rec.append(r)
                f.append(f1(p, r))
            average_array[i] = [sum(acc)/n_seed,sum(prec)/n_seed,sum(rec)/n_seed,sum(f)/n_seed]

        averages = pd.DataFrame(average_array, columns=['Average Accuracy','Average Precision','Average Recall','Average F1'])
        final_df = pd.concat([df.iloc[:,range(df.shape[1]-n_folds)], averages],axis=1)
        final_df.to_csv(filename+'score.csv',index=True)











    
