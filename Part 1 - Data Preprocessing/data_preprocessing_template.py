# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing, model_selection #import Imputer, LabelEncoder, OneHotEncoder

# Load data into pd dataframe
df = pd.read_csv('Data.csv')

# Convert pd dataframe into np array
data = df.values

# X and y values (features and labels)
X = data[:, 0:-1]
y = data[:, -1]

# Data preprocessing- filling missing values
imputer = preprocessing.Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3]).astype(float)

# Encode categorical data

#1. Encode the features
labelencoder_X = preprocessing.LabelEncoder()
#LabelEncoder does Integer encoding of the categorical data \
#Integer encoding retains a natural ordering (ordinal relationship) \
#0<1<2...<n which some ML algorithms might want to harness. This would \
#not be undesirable for cateogorical data which do not have any ordinal relationship \
#for eg, Country categorical variable (variable that stores categorical data) \
#has values that have no such ordinal relationship. LabelEncoder, used below, will \
#do an integer encoding and the return values therefore will have ordinal relationship.
#To avoid values from having ordinal relationship, we use OneHotEncoder

#using LabelEncoder obj--avoid using this class if u don't want ordinality
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
#alternatively
#labelencoder_X = labelencoder_X.fit(X[:,0])
#X[:, 0] = labelencoder_X.transform(X[:,0])

#using OneHotEncoder-- use this if you want conversion which doesn't inherit ordinality
onehotencoder_X = preprocessing.OneHotEncoder(categorical_features = [0])
X = onehotencoder_X.fit_transform(X).toarray()

#2. Encode the y-labels
#no need to apply OneHotEncoder on y, because this is the dependent variable, ie, \
#labels to the features. ML algos don't look for ordinality on dependent variable \
#values, they do so only for features.
labelencoder_Y = preprocessing.LabelEncoder()
labelencoder_Y = labelencoder_Y.fit(y)
y = labelencoder_Y.transform(y)

#3. Split dataset into training and test
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2, random_state = 0)