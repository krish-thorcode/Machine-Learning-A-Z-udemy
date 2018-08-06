# Random Forest Classification

# Importing the modules
import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import preprocessing, model_selection, ensemble
from sklearn.metrics import confusion_matrix

df = read_csv('Social_Network_Ads.csv')

X= df.iloc[:, [2,3]].values
y = df.iloc[:, -1].values

# Data Splitting
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y,\
                                        test_size= 0.25, random_state= 0)

# Feature scaling (although random forest does not use eucledian distance\
# or kernel anywhere in its implementation, feature scaling is being done to \
# facilitate the plotting of the countours for decision boundary because\
# np.meshgrid consumes a lot of RAM, w/o feature scaling, I was getting Memory\
# error although I have 8 GB of ram)
scaler_X = preprocessing.StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test= scaler_X.transform(X_test)

# Random Forest Classification model fitting
# default value of number of trees in the forest, n_estimators = 10,\
# criterion = 'gini'
classifier= ensemble.RandomForestClassifier(n_estimators = 10,\
                                            criterion = 'entropy',\
                                            random_state = 0)
classifier= classifier.fit(X_train, y_train)

y_predictions = classifier.predict(X_test)

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_predictions)

# Visualising decision boundary with scattered training set
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(X_set[:, 0].min() - 1, X_set[:, 0].max() + 1,\
                               step = 0.01),
                    np.arange(X_set[:, 1].min() - 1, X_set[:, 1].max() + 1,\
                              step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).\
            reshape(X1.shape), alpha = 0.4,\
            cmap = ListedColormap(('red', 'green')))
for i,j in enumerate(np.unique(y_test)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], s = 3, label = j,\
                c = ListedColormap(('red','green'))(j))
    
plt.title('Random Forest(training set)')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.show()

# Visualising decision boundary with scattered test set
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(X_set[:, 0].min() - 1, X_set[:, 0].max() + 1,\
                               step = 0.01),
                    np.arange(X_set[:, 1].min() - 1, X_set[:, 1].max() + 1,\
                              step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).\
            reshape(X1.shape), alpha = 0.4,\
            cmap = ListedColormap(('red', 'green')))
for i,j in enumerate(np.unique(y_test)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], s = 3, label = j,\
                c = ListedColormap(('red','green'))(j))
    
plt.title('Random Forest(training set)')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.show()