# Decision Tree Regression

#import modules
import numpy as np
from pandas import read_csv
from sklearn import preprocessing, model_selection, tree
import matplotlib.pyplot as plt

df = read_csv('Position_Salaries.csv')

X = df.iloc[:, 1:2].values
y = df.iloc[:, -1].values

#Regressor (Decision tree regressor)
regressor = tree.DecisionTreeRegressor(random_state = 0)
regressor = regressor.fit(X, y)

y_predictions = regressor.predict(X)

y_single_prediction = regressor.predict(6.5)
# Plotting
#The plot below is not the best way a decision tree can be represented because \
#we're plotting predicted y values for few discrete values of X each of which fall \
#in different splits of the independent variable. Recall that within a particular \
#split the prediction will be same for all values of independent variable values \ 
#which is equal to the average of the dependent variable values corresponding to
#that split. So, within a split we should get a flat line as prediction will be \
#same for any independent variable (within that split only). Hence, we try to plot the result such a way \
#that the we get atleast two independent variables within a given split to show \
#flat line within that split.

#plt.scatter(X, y, color = 'red')
#plt.title('Decision tree: Truth or Bluff?')
#plt.xlabel('Level')
#plt.ylabel('Salary')
#plt.plot(X, y_predictions, color = 'blue')
#plt.show()
#Following does the job
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = 'red')
plt.title('Decision tree: Truth or Bluff?')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.show()