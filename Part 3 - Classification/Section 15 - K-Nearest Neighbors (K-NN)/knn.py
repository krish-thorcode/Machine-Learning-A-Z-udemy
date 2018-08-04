# K-Nearest Neighbors (K-NN)

# Importing modules
import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
import matplotlib.colors
from sklearn import preprocessing, model_selection, neighbors, metrics

df = read_csv('Social_Network_Ads.csv')
X = df.iloc[:, [2,3]].values
y = df.iloc[:, -1].values

# split data into train, test sets because size of dataset allows us to do so
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y,\
                                        test_size = 0.25, random_state = 0)

# Feature Scaling- KNN (KNeighborsClassifier) doesn't do scaling on its own
scaler_X = preprocessing.StandardScaler()
scaler_X = scaler_X.fit(X)
X_train = scaler_X.transform(X_train)
X_test = scaler_X.transform(X_test)

# Fitting KNearestNeighbour model
#default value of n_neighbors = 5, metric = 'minkowski' (for eucledian disance),\
#p = 2
classifier = neighbors.KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski',\
                                          p = 2)
classifier = classifier.fit(X_train, y_train)

y_predictions = classifier.predict(X_test)

conf_matrix = metrics.confusion_matrix(y_test, y_predictions)
print(conf_matrix)

# Visualising training set with the prediction boundary that separates two\
#  classes learned by KNeighborsClassifier during training 
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(X_set[:, 0].min() - 1, X_set[:, 0].max() + 1,\
                    step = 0.01),\
                    np.arange(X_set[:, 1].min() - 1, X_set[:, 1].max() + 1, \
                    step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).\
             reshape(X1.shape),alpha = 0.6, cmap = matplotlib.colors.\
             ListedColormap(('red','green')))

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_train == j, 0], X_set[y_train == j, 1],\
                c = matplotlib.colors.ListedColormap(('red', 'green'))(i),\
                label = j)
    
plt.title('K-NN (training)')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.show()

#Visualising the prediction boundary and the test examples
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(X_set[:, 0].min() - 1, X_set[:, 0].max() + 1,\
                    step = 0.01),\
                    np.arange(X_set[:, 1].min() - 1, X_set[:, 1].max() + 1, \
                    step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).\
             reshape(X1.shape),alpha = 0.6, cmap = matplotlib.colors.\
             ListedColormap(('red','green')))

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],\
                c = matplotlib.colors.ListedColormap(('red', 'green'))(i),\
                label = j)
    
plt.title('K-NN (test)')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.show()