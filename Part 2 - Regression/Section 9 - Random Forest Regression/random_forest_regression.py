# Random Forest Regression

#Importing modules
import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
from sklearn import ensemble

df = read_csv('Position_Salaries.csv')
X = df.iloc[:, 1:2].values
y = df.iloc[:, -1].values

#Random forest regressor
#n_estimators is the number of trees in RandomForestRegressor obj, default = 10
randomforest_regressor = ensemble.RandomForestRegressor(n_estimators = 300,\
                                                        random_state = 0)
randomforest_regressor = randomforest_regressor.fit(X, y)
y_predictions = randomforest_regressor.predict(X)

y_prediction_single = randomforest_regressor.predict(6.5)

#Plotting the dataset and output for the Random Forest (also Decision Tree) \
#the best way
X_grid = np.arange(min(X), max(X), 0.01) # increase resolution by decreasing last parameter val
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, randomforest_regressor.predict(X_grid), color = 'blue')
plt.show()