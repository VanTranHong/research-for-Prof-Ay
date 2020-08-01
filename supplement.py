import data_preprocess as dp
import numpy as np
import pandas as pd
import scoring as score
import normal_run as nr
import normal_run_parallel_ranked as parallel
import featureselection as fselect
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import  SGDClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import statsmodels.api as sm
import scoring as sr


##### reading from csv file that the summary of confusion matrix, will calculate accuracy and F1 score for each matrix and write in 2 files one for accuracy and 1 for F1 score
data = pd.read_csv('SUMMARY.csv',skipinitialspace=True, header = 0)
data1 = data.iloc[:,[5,6,7,8,9]]
numpy_arr = data1.to_numpy()
# print(type(numpy_arr))
num_row = data1.shape[0]
rowsaccuracy = []
rowsf1=[]
for i in range(num_row):
    row_ac =[]
    row_f1 =[]
    for j in range(5):
        str = data1.iloc[i,:][j]
        
        new_str=str[1:len(str)-1]
        lst = new_str.split()
        tn = int(lst[0])
        fp = int(lst[1])
        fn = int(lst[2])
        tp = int(lst[3])
        row_ac.append(sr.accuracy(tp, tn, sum([tn,fp,fn,tp])))
        p = sr.precision(tp, fp)
        r = sr.recall(tp, fn)
        row_f1.append(sr.f1(p, r))
    rowsaccuracy.append(row_ac)
    rowsf1.append(row_f1)
accuracies = pd.DataFrame(rowsaccuracy, columns=['Accuracy1','Accuracy2','Accuracy3','Accuracy4','Accuracy5'])
accuracies.to_csv('accuracy_score_in_summary.csv',index=True)
f1s= pd.DataFrame(rowsf1, columns=['F1_1','F1_2','F1_3','F1_4','F1_5'])
f1s.to_csv('f1_score_in_summary.csv',index=True)

    

    