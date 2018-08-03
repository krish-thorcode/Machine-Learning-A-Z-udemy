# Polynomial regression
import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
from sklearn import preprocessing, linear_model#, model_selection

df = read_csv('Position_Salaries.csv')
# On seeing the dataframe we see that the data 'Position' and 'Level' convey \
# basically identical informations and thus, keeping both will be redundant \
# which is not a healthy practice. So, we take only the level feature into X \
# because that is basically a numerical (ordinal) representation of Position.

#X = df.iloc[:, 1].values #this will return a vector of features \
#while building ML models, we want the features to be in matrix format, not \
#as vectors- this is how the standard convention goes.\
#So, we do the following instead. Alternatively we could also use reshape() method.
X = df.iloc[:,1:2]
y = df.iloc[:, -1].values # we want y to be a vector

# not going to split our dataset this time..because the dataset is small \
# and the company's method of entitling salaries seems to be a pretty solid one \
# ie, the fitting is required to be done as perfectly as possible for this case \
# we use all the data to train our model. so, the split will not be done this time.
#X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y,\
#                                        test_size = 0.2, random_state = 0)

# No need to do feature scaling because the LinearRegression class does this \
# on its own, and we are going to use the same class to make a polynomial regr \
# model which is still a linear model because the parameters are linear.
poly_feature_creator = preprocessing.PolynomialFeatures(degree = 2)
# the returned matrix from poly_feature_cretator will also have intercepts
X_poly = poly_feature_creator.fit_transform(X)
regressor = linear_model.LinearRegression()
regressor = regressor.fit(X_poly, y)

y_predictions = regressor.predict(X_poly)

plt.scatter(X, y, color = 'red')
plt.plot(X, y_predictions, color = 'blue')

# training a linear regression model just for comparision
linear_regressor = linear_model.LinearRegression()
linear_regressor = linear_regressor.fit(X, y)
y_linearpred = linear_regressor.predict(X)
plt.plot(X, y_linearpred, color = 'orange')

#training a 3-degree polynomial model 
poly_feature_creator = preprocessing.PolynomialFeatures(degree = 3)
X_poly = poly_feature_creator.fit_transform(X)
regressor = linear_model.LinearRegression()
regressor = regressor.fit(X_poly, y)
y_predictions = regressor.predict(X_poly)
plt.plot(X, y_predictions, color = 'green')

#training a 4-degree polynomial model
poly_feature_creator = preprocessing.PolynomialFeatures(degree = 4)
X_poly = poly_feature_creator.fit_transform(X)
regressor = linear_model.LinearRegression()
regressor = regressor.fit(X_poly, y)
y_predictions = regressor.predict(X_poly)
plt.plot(X, y_predictions, color = 'black')
plt.show()

# polynomial model with degree = 4 is selected
# predicting for only one value, lets say 6.5
#1. prediction by linear_regressor
linear_regressor.predict(6.5)

#2. prediction by polynomial (degree 4) regressor
regressor.predict(poly_feature_creator.fit_transform(6.5))