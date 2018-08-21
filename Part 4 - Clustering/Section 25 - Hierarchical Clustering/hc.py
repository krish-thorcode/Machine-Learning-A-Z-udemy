# Hierarchical Clustering

# Importing modules
import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

df = read_csv('Mall_Customers.csv')

X = df.iloc[:, [3,4]].values

# Optimal number of clusters using Dendrogram
# sch.linkage is the agglomerative hc algorithm itself. linkage is used for \
# the processes of forming clusters (links) which is done in this algorithm.\
# method is the method using which the linkage (clusters) will be formed
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.xlabel('Customers') # not that x axis is not one of the 2 features in the dataset\
                        # that we are using. x axis consists of what the each pair of\
                        # features represent conceptually..in this case each data point\
                        # represents a customer
plt.ylabel('Euclidean distance/Degree of dissimilarity') # y-axis is the measure of how\
                        # dissimilar two datapoints (which depict two customers) are \
                        # which is directly based on the euclidean distance
plt.title('Dendogram')
plt.show()

# From the dendrogram, optimum number of clusters = 5 (same as found using elbow method)\
# in K-means clustering
# Fitting the Agglomerative Hierarchical Clustering model
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X) # we can use fit and predict separately as well, \
                        # one after the other

# Visualising the clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 60, c = 'red', label = 'Cluster 1',\
            edgecolors = 'black')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 60, c = 'blue', label = 'Cluster 2',\
            edgecolors = 'black')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 60, c = 'green', label = 'Cluster 3',\
            edgecolors = 'black')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 60, c = 'cyan', label = 'Cluster 4',\
            edgecolors = 'black')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 60, c = 'magenta', label = 'Cluster 5',\
            edgecolors = 'black')
plt.legend()
plt.title('Cluster of customers')
plt.xlabel('Annual income of customers (k $)')
plt.ylabel('Spending score(1-100')
plt.show()