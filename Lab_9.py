import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Setting seaborn style
sns.set()

# Loading the data
data = pd.read_csv('3.01. Country clusters.csv')

# Plotting the data
plt.scatter(data['Longitude'], data['Latitude'])
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Plot of Data')
plt.show()

# Selecting the feature
x = data.iloc[:, 1:3]

# K-Means Clustering
kmeans = KMeans(n_clusters=3)
kmeans.fit(x)

# Clustering results
identified_clusters = kmeans.predict(x)
data_with_clusters = data.copy()
data_with_clusters['Clusters'] = identified_clusters

# Plotting the clustered data
plt.scatter(data_with_clusters['Longitude'], data_with_clusters['Latitude'], c=data_with_clusters['Clusters'], cmap='rainbow')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Clustered Data')
plt.show()

# WCSS and Elbow Method
wcss = []
for i in range(1, 7):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(x)
    wcss_iter = kmeans.inertia_
    wcss.append(wcss_iter)

# Plotting the Elbow Method
number_clusters = range(1, 7)
plt.plot(number_clusters, wcss, marker='o', linestyle='-')
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.show()
