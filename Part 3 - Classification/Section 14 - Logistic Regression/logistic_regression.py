# Logistic Regression

# Importing modules
import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
import matplotlib.colors
from sklearn import preprocessing, model_selection, linear_model, metrics

df = read_csv('Social_Network_Ads.csv')

# Checking the df variable content shows a number of columns that could be used \
# as features for the problem, but a business decision has been made that the only\
# age and estimated salary should be taken as features
X = df.iloc[:, [2,3]].values
y = df.iloc[:, -1].values

# Splitting dataset because the dataset is big enough for such a split (400 egs)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y,\
                                                test_size = 0.25, random_state = 0)

# Feature scaling, LogisticRegression doesn't do it unlike LinearRegression class
scaler_X = preprocessing.StandardScaler()
scaler_X = scaler_X.fit(X_train)
X_train = scaler_X.transform(X_train)
X_test = scaler_X.transform(X_test)

# Fitting LogisticRegression model
classifier = linear_model.LogisticRegression()
classifier = classifier.fit(X_train, y_train)

# Predictions on test set
y_predictions = classifier.predict(X_test)

# Making the Confusion Matrix for evaluating the logistic regression predictions

conf_matrix = metrics.confusion_matrix(y_test, y_predictions)
print(conf_matrix)

#Visualising training data set and decision boundary
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(X_set[:, 0].min() - 1, X_set[:, 0].max() + 1,\
                         step = 0.01),\
                    np.arange(X_set[:, 1].min() - 1, X_set[:, 1].max() + 1, \
                        step = 0.01))

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).\
             reshape(X1.shape), alpha = 0.75, cmap = matplotlib.colors.\
             ListedColormap('red', 'green'))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],\
                c = matplotlib.colors.ListedColormap(('red','green'))(i), label = j)