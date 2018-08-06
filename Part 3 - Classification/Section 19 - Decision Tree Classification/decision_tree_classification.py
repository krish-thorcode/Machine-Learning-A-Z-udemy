# Decision Tree Classification

# Importing modules
import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
from sklearn import preprocessing, model_selection, tree
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap

df = read_csv('Social_Network_Ads.csv')
X = df.iloc[:, [2,3]].values
y = df.iloc[:, -1].values

# Dataset splitting into train and test
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, \
                                        test_size = 0.25, random_state = 0)

# Feature scaling
# Decision tree classifier is not a Eucluedian distance based algorithm, so\
# feature scaling is not requird since the algorithm compares the values, and\
# does not computes any kind of kernel or distance. However, for plotting purpose,\
# you may want to do feature scaling. We'll try both- plotting w/o feature scaling\
# and plotting with feature scaling.
# Result: Running the plot-code snippet without feature scaling results in\
# memory error, because the plot-code uses np.meshgrid which consumes whose output\
# consumes a lot of memory. Using large numbers further amplifies this memory\
# requirement, which caused 'Memory error' in my system which has an 8 GB of RAM.
scaler_X = preprocessing.StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

# Fitting Decision Tree Classifier
classifier = tree.DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier = classifier.fit(X_train, y_train)

y_predictions = classifier.predict(X_test)

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_predictions)

# Visualising the decision boundary with training set
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(X_set[:, 0].min() - 1, X_set[:, 0].max() + 1,\
                         step = 0.01),\
                    np.arange(X_set[:, 1].min() - 1, X_set[:, 1].max() + 1, \
                        step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).\
             reshape(X1.shape), alpha = 0.5,\
             cmap = ListedColormap(('red','green')))

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],\
                c =ListedColormap(('red', 'green'))(j), label = j, s = 3)

plt.title('Decision tree classification (Training set)')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

# Visualising the decision boundary with test set
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(X_set[:, 0].min() - 1, X_set[:, 0].max() + 1,\
                         step = 0.01),\
                    np.arange(X_set[:, 1].min() - 1, X_set[:, 1].max() + 1, \
                        step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).\
             reshape(X1.shape), alpha = 0.5,\
             cmap = ListedColormap(('red','green')))

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],\
                c =ListedColormap(('red', 'green'))(j), label = j, s = 3)

plt.title('Decision tree classification (Training set)')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

