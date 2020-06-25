import pandas as pd 
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def remove_invalid_column(data):
    """Function that removes columns that aren't suitable for machine learning.
    This includes features with more than 5% missing values, wrong data type,
    and the indices.

    Args:
        data (Pandas DataFrame): The DataFrame that contains data that hasn't been preprocessed.

    Returns:
        DataFrame: Preprocessed DataFrame.
    """
    data = data[data.columns[data.isnull().mean()<0.05]]
    data = data.select_dtypes(exclude=['object'])
    data = data[data['HPYLORI'].notna()]
    data = data.drop('Unnamed: 0', axis=1)
    return data

def recategorize(data):
    """Recategorizes the data according to the standards in the SPSS file.

    Args:
        data (DataFrame): DataFrame containing data that hasn't been preprocessed.

    Returns:
        DataFrame: A DataFrame with replaced values.
    """
    data["ADD"].replace({2: 0}, inplace=True)
    data["AnyPars3"].replace({2: 0, 9: 0}, inplace=True)
    data["CookArea"]=data["CookArea"]-1
    data["GCOW"]=data["GCOW"]-1 
    data["DEWORM"].replace({0: 'a', 1: 0}, inplace=True)
    data["DEWORM"].replace({'a': 1}, inplace=True) 
    data["GCAT"].replace({1: 0, 2: 1, 3: 2}, inplace=True)
    data["GDOG"].replace({1: 0, 2: 1, 3: 2}, inplace=True)
    data["GELEC"].replace({3: 0, 2: 'a', 1: 2}, inplace=True)
    data["GELEC"].replace({'a': 1}, inplace=True)
    data["GFLOOR6A"].replace({1: 0, 2: 1, 3: 2}, inplace=True)
    data["GWASTE"].replace({4: 0}, inplace=True)
    data["HCIGR6A"].replace({1: 0, 2: 1}, inplace=True)
    data["HPYLORI"].replace({1: 0, 2: 1}, inplace=True)
    data["AgeGroups"].replace({1: 0, 2: 1, 3: 2}, inplace=True)
    data["FamilyGrouped"].replace({1: 0, 2: 1, 3: 2}, inplace=True)
    data["ToiletType"].replace({1: 0, 2: 1, 3: 2}, inplace=True)
    data["WaterSource"].replace({1: 0, 2: 1, 3: 2}, inplace=True)
    return data

def getnominal(data):
    """Finds the features that contain nominal values.

    Args:
        data (DataFrame): DataFrame containing the dataset.

    Returns:
        List: A list that contains the nominal features.
    """
    nominal = []
    for col in data.columns:
        distinct = np.sort(data[col].dropna().unique())
        if len(distinct) > 2:
            nominal.append(col)    
    return nominal

def create_dummies (data):
    """Creates dummy variables.

    Args:
        data (DataFrame): DataFrame containing the dataset

    Returns:
        DataFrame: DataFrame containing the dataset with dummy variables.
    """
    dummy = pd.get_dummies(data, columns = data.columns, drop_first= True) 
    return dummy

def impute(data):
    """Multivariate imputer that estimates each feature from all the others

    Args:
        data (Numpy Array): A numpy array containing the dataset.

    Returns:
        Numpy Array: A matrix with imputed values.
    """
    imputed_data = []
    for i in range(5):
        imputer = IterativeImputer(sample_posterior=True, random_state=i)
        imputed_data.append(imputer.fit_transform(data))
    returned_data = np.round(np.mean(imputed_data,axis = 0))
    
    return returned_data

def modify_data(data, numerical, nominal):
    """Runs all the preprocessing functions on the dataset.

    Args:
        data (DataFrame): DataFrame containing the dataset with no preprocessing.
        numerical (List): List containing all the features that are ordinal.
        nominal (List): List containing all the features that are nominal.

    Returns:
        DataFrame: DataFrame with all the preprocessing done.
    """
    data = remove_invalid_column(data)
    data = recategorize(data)
    columns = data.columns
    data.columns = columns   
    nominal = getnominal(data)  
  
    nominal_data = create_dummies (data[nominal])
    data = data.drop(nominal, axis = 1)
    data = pd.concat([data,nominal_data], axis =1)
   
    return data