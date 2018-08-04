#SVR

#importing modules
import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
from sklearn import preprocessing, model_selection, svm

df = read_csv('Position_Salaries.csv')

X = df.iloc[:, 1:2].values
y = df.iloc[:, -1].values

#LinearRegression class took care of feature scaling on its own, but SVR doesnt

#Split into train and tst data sets
#not doing it for this problem
#X_train, y_train, X_test, y_test = model_selection.train_test_split(\
#                                        X, y, test_size = 0.2,random_state = 0)


# without feature scaling, the output will be very bad
scaler_X = preprocessing.StandardScaler()
scaler_X = scaler_X.fit(X)
X = scaler_X.transform(X)
scaler_y = preprocessing.StandardScaler()
#scaler_y = scaler_y.fit(y)
y = scaler_y.fit_transform(y.reshape(y.shape[0], 1)) # accepts 2D array

# Create SVR regressor
svr_regressor = svm.SVR(kernel = 'rbf') # default is rbf, no need to specify \
                                        # if we want to use Gaussian

svr_regressor.fit(X, y)
y_predictions = svr_regressor.predict(X)

#Predict for a single value
#Since the regressor has been fitted to scaled fetures, we need to give scaled \
#inputs to the unknown test data as well
new_input = 6.5
scaled_new_input = scaler_X.transform(6.5)
y_predict_single = svr_regressor.predict(scaled_new_input)
y_predict_single = scaler_y.inverse_transform(y_predict_single)
print(y_predict_single)
# Plotting the data and results
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, svr_regressor.predict(X_grid), color = 'blue')
plt.title('SVR- Truth or Bluff')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()