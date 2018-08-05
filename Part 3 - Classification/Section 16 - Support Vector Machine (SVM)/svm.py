# Support Vector Machine (SVM)

# Import modules
import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
import matplotlib.colors
from sklearn import preprocessing, model_selection, svm, metrics

df = read_csv('Social_Network_Ads.csv')
X = df.iloc[:, [2,3]].values
y = df.iloc[:, -1].values

# Splitting into training and test sets as the dataset is big enough
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y,\
                                        test_size = 0.25, random_state = 0)

# Feature scaling
scaler_X = preprocessing.StandardScaler()
scaler_X = scaler_X.fit(X_train)
X_train = scaler_X.transform(X_train)
X_test = scaler_X.transform(X_test)

# SVM Model fitting
#default value for kernel is 'rbf' which is Gaussian kernel
classifier = svm.SVC(kernel = 'linear', random_state = 0)
classifier = classifier.fit(X_train, y_train)
y_predictions = classifier.predict(X_test)

# Confusion matrix
conf_matrix = metrics.confusion_matrix(y_test, y_predictions)
print(conf_matrix)

#Visualisation decision boundary, ie, Hyperplane and with training set scattered
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(X_set[:, 0].min() - 1, X_set[:, 0].max() + 1,\
                     step = 0.01),
                     np.arange(X_set[:, 1].min() - 1, X_set[:, 1].max() + 1,\
                     step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).\
           reshape(X1.shape), alpha = 0.6,\
           cmap = matplotlib.colors.ListedColormap(('red','green')))

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],\
                c = matplotlib.colors.ListedColormap(('red', 'green'))(i),\
                label = j, s = 3)
plt.title('SVM (Training set)')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.show()

# Visualisation decision boundry or Hyperplane with the test set scattered
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(X_set[:, 0].min() - 1, X_set[:, 0].max() + 1,\
                               step = 0.01),\
                    np.arange(X_set[:, 1].min() - 1, X_set[:, 1].max() + 1,\
                              step = 0.01))
    
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).\
             reshape(X1.shape), alpha = 0.6,\
             cmap = matplotlib.colors.ListedColormap(('red', 'green')))

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_test[y_test == j, 0], X_test[y_test == j, 1],\
                c = matplotlib.colors.ListedColormap(('red', 'green'))(j),\
                label = j, s = 3)
plt.title('SVM(Test set)')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.show()