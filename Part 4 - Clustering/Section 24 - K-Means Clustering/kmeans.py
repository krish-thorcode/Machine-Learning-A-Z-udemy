# importing libraries

import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = read_csv('Mall_Customers.csv')

X = df.iloc[:, [3,4]].values

# Use elbow method to find the optimal number of clusters
wcss = []
for i in range(1, 11):
    #n_clusters = number of clusters for K-Means,\
    #init = random initalisation method to avoid random initialisation trap\
    #max_iter = maximum number of iterations for one initial centroid\
    #n_init = number of different centroids for the same clustering attempt, mtlb\
    #pehla baar centroid choose kia...phr 300 iteration tk jo cluster aaya..\
    #ab dusra baar dusra centroid choose karo..phr dekho cluster kya aaya..\
    #aisa 10 baar karo..phr best cluster jo aaya hoga inn 10 baar me..\
    #wo wala clustering result rakho..\
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300,\
                    n_init = 10, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot the graph for wcss to find the elbow
plt.plot(range(1,11), wcss, marker = 'o')
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# From the elbow method we get the optimal number of clusters which is 5 for this\
# problem

# Applying KMeans with the number of clusters determined
kmeans = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10,\
                random_state = 0)
#y_kmeans = kmeans.fit_predict(X); -- this is equivalent to doing the following
kmeans.fit(X);
y_kmeans = kmeans.predict(X);
