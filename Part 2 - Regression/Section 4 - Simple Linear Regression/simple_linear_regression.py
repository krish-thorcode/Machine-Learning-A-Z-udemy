# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing, model_selection, linear_model

# Load data into pd dataframe
df = pd.read_csv('Salary_Data.csv')

# Convert pd dataframe into np array
data = df.values

# X and y values (features and labels)
X = data[:, 0:-1]
y = data[:, -1]

# Encode categorical data

#1. Encode the features

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 1/3, random_state = 0)

"""scaler_X = preprocessing.StandardScaler()
scaler_X = scaler_X.fit(X_train)
X_train = scaler_X.transform(X_train)
X_test = scaler_X.transform(X_test)"""

# LinearRegression object
linear_regressor = linear_model.LinearRegression()
linear_regressor = linear_regressor.fit(X_train, y_train)

y_predictions = linear_regressor.predict(X_test)

# Visualising the training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, linear_regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

#Visualising the test set performance
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_test, y_predictions, color = 'blue') # note that this line is the \
        #same as line plotted on X_train and linear_regressions.predict(X_train) \
    #because y_predictions are computed by using the same line, X_test is substituted \
        #on the same line
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
