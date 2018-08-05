# Kernel SVM

#Importing modules
import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import preprocessing, model_selection, svm , metrics

df = read_csv('Social_Network_Ads.csv')
X = df.iloc[:, [2,3]].values
y = df.iloc[:, -1].values

# Splitting data
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y,\
                                        test_size = 0.25, random_state = 0)

# Feature scaling
scaler_X = preprocessing.StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

# Training the Kernel SVM model
classifier= svm.SVC(kernel = 'rbf', random_state = 0)
classifier = classifier.fit(X_train, y_train)

y_predictions = classifier.predict(X_test)

# Confusion matrix
conf_matrix = metrics.confusion_matrix(y_test, y_predictions)
print(conf_matrix)

# Visualise Training data with the decision boundary learned during training
X_set , y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(X_set[:, 0].min() - 1, X_set[:, 0].max() + 1,\
                     step = 0.01),\
                     np.arange(X_set[:, 1].min() - 1, X_set[:, 1].max() + 1,\
                    step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).\
             reshape(X1.shape), alpha = 0.6,\
             cmap = ListedColormap(('red', 'green')))

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],\
                c = ListedColormap(('red', 'green'))(j), s = 3, label = j)
plt.title('SVM (RBF kernel)')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.show()

# Visualise test data with the decision boundary
X_set , y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(X_set[:, 0].min() - 1, X_set[:, 0].max() + 1,\
                     step = 0.01),\
                     np.arange(X_set[:, 1].min() - 1, X_set[:, 1].max() + 1,\
                    step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).\
             reshape(X1.shape), alpha = 0.6,\
             cmap = ListedColormap(('red', 'green')))

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],\
                c = ListedColormap(('red', 'green'))(j), s = 3, label = j)
plt.title('SVM (RBF kernel)')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.show()