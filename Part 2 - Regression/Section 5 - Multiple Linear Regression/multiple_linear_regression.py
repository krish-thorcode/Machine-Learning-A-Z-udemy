#Data preprocessing

import numpy as np
from pandas import read_csv
from sklearn import preprocessing, model_selection, linear_model
import statsmodels.formula.api as sm

df = read_csv('50_Startups.csv')

# df.values returns a numpy array for the dataframe object. In this case, df.values \
# returns a numpy array of objects because all the datatypes are not same inside the df \
# however, we know that y consists of all float values and anyway we need to extract it \
# as a separate vector. Doing data = df.values first and then extracting X and y \
# would mean we are extracting objects from the data variable as it is an array of \
# objects. this does not seem to be an optimal thing to do, rather if we extract \
# X and y values separately from df directly, then it would return an array of objects \
# for X and a vector of floats for y. This can be done using iloc attribute of dataframe \
# object. Recall that c is called intercept.
#data = df.values
#X = data[:, 0:-1]
#y = data[:, -1]

# Extracting X and y directly from the dataframe object df, so that the y is directly \
# extracted as a vector of floats.\
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

labelencoder_X = preprocessing.LabelEncoder()
#X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
labelencoder_X = labelencoder_X.fit(X[:, 3])
X[:, 3] = labelencoder_X.transform(X[:, 3])

onehotencoder_X = preprocessing.OneHotEncoder(categorical_features = [3])
#X = onehotencoder_X.fit_transform(X).toarray()
onehotencoder_X = onehotencoder_X.fit(X)
X = onehotencoder_X.transform(X).toarray()

#AVoiding dummy variable trap
#although the LinearRegression library that we are going to use does this step for us \
#still writing it just as a reminder that this is an actual step that should be  \
# kept in mind while coding a model
X = X[:, 1:]

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y,\
                                        test_size = 0.2, random_state = 0)


#Feature scaling is not required to be done manually because the LinearRegression \
#library does this for us

#Fitting multiple linear regression model, ie, multivariate linear regression model
linear_regressor = linear_model.LinearRegression()
linear_regressor = linear_regressor.fit(X_train, y_train)

y_predictions = linear_regressor.predict(X_test)

# Building the optimal model using Backward Elimination
#we shall be using the statsmodels.formula.api module to find p-values and then \
#use it to find the optimal multivariate linear regression model
#this model does not account for the presence of the constant term in the linear \
#model, ie, model won't be knowing that there should be a 'c' in the y = mx+c \
#equation. We have to add a vector of ones to our feature matrix so that the module \
#will be doing the things as per our expectations. LinearRegression class, on the other, \
#hand, does this on our behalf, ie, it accounts for a constant 'c' by default.

ones = np.ones(shape = (X.shape[0], 1))
X = np.append(ones, X, axis = 1)
X_opt = X[:, [0, 1, 2, 3, 4, 5]] # writing indices this way so that \
                                # manully removing features becomes easier
regressor_ols = sm.OLS(endog = y, exog = X_opt)
regressor_ols = regressor_ols.fit() # fit the model with all possible predictors (features)
#using OLS regressor because of summary() method in sm that shows p-values as well
regressor_ols.summary()

X_opt = X[:, [0, 1, 3, 4, 5]] # removed 2nd feature which had highest p-value
regressor_ols = sm.OLS(endog = y, exog = X_opt).fit()
regressor_ols.summary()

X_opt = X[:, [0,3,4,5]]
regressor_ols  = sm.OLS(endog = y, exog = X_opt).fit()
regressor_ols.summary()

X_opt = X[:, [0,3,5]]
regressor_ols = sm.OLS(endog = y, exog = X_opt).fit()
regressor_ols.summary()