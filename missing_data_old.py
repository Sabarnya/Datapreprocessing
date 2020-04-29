# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import sklearn
from sklearn.impute import SimpleImputer


# Importing the dataset
dataset = pd.read_csv('G:\Life at University of Paderborn\Extra curricular studies\Machine learning\P14-Data-Preprocessing\Data_Preprocessing\Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

#print(sklearn.__version__)

# Taking care of missing data
#from sklearn.preprocessing import Imputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])