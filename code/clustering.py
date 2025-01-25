"""from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
import pandas as pd
import os
import sys


# Load data
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
alloy_dataset = pd.read_csv(project_path + "\\data\\refinedData\\alloy.csv")

# Assuming `alloy_dataset` is your dataset with a 'Tc' column
tc_values = alloy_dataset['critical_temp'].values.reshape(-1, 1)



# Fit K-Means
kmeans = KMeans(n_clusters=5, random_state=42)
alloy_dataset['Cluster'] = kmeans.fit_predict(tc_values)



# Assign labels to clusters (e.g., Low, Medium, High)
cluster_centers = kmeans.cluster_centers_.flatten()
cluster_labels = ['Low', 'Medium-Low', 'Medium', 'Medium-High', 'High']  # Adjust based on the number of clusters
sorted_clusters = sorted(zip(cluster_centers, cluster_labels))
cluster_mapping = {i: label for i, (center, label) in enumerate(sorted_clusters)}

alloy_dataset['Goodness'] = alloy_dataset['Cluster'].map(cluster_mapping)


def classify_alloy(tc, kmeans, cluster_mapping):
    cluster = kmeans.predict([[tc]])[0]
    return cluster_mapping[cluster]

# Example
new_tc = 15.0  # Critical temperature of a new alloy
goodness = classify_alloy(new_tc, kmeans, cluster_mapping)
print(f"The alloy is classified as: {goodness}")



from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

distortions = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(tc_values.reshape(-1, 1))
    distortions.append(kmeans.inertia_)

plt.plot(range(1, 10), distortions, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Distortion')
plt.title('Elbow Method')
plt.show()

import matplotlib.pyplot as plt
plt.scatter(tc_values, [0] * len(tc_values), c=alloy_dataset['Cluster'], cmap='viridis')
plt.scatter(kmeans.cluster_centers_, [0] * len(kmeans.cluster_centers_), c='red', marker='x')
plt.xlabel('Critical Temperature (Tc)')
plt.title('Clustering of Alloys Based on Tc')
plt.show()"""
import numpy as np
import pandas as pd
import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load data
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
alloy_dataset = pd.read_csv(project_path + "\\data\\refinedData\\alloy.csv")

# Remove the element(s) with the highest critical temperature
max_critical_temp = alloy_dataset['critical_temp'].max()
alloy_dataset = alloy_dataset[alloy_dataset['critical_temp'] < max_critical_temp]

# Update tc_values after removing the outlier
tc_values = alloy_dataset['critical_temp'].values.reshape(-1, 1)

# Fit K-Means
kmeans = KMeans(n_clusters=5, random_state=42)
alloy_dataset['Cluster'] = kmeans.fit_predict(tc_values)

# Assign labels to clusters (e.g., Low, Medium, High)
cluster_centers = kmeans.cluster_centers_.flatten()
cluster_labels = ['High', 'Medium-High', 'Medium', 'Medium-Low', 'Low']  # Adjust based on the number of clusters
sorted_clusters = sorted(zip(cluster_centers, cluster_labels))
cluster_mapping = {i: label for i, (center, label) in enumerate(sorted_clusters)}

alloy_dataset['Goodness'] = alloy_dataset['Cluster'].map(cluster_mapping)


def classify_alloy(tc, kmeans, cluster_mapping):
    # Get the predicted cluster
    cluster = kmeans.predict([[tc]])[0]
    
    # Check if the temperature is above the highest cluster center
    if tc > max(cluster_centers):
        return 'High'
    
    # Otherwise, return the mapped label
    return cluster_mapping[cluster]


# Example
new_tc = 1500.0  # Critical temperature of a new alloy
goodness = classify_alloy(new_tc, kmeans, cluster_mapping)
print(f"The alloy is classified as: {goodness}")
"""
# Visualize the clustering after removing the outlier
plt.scatter(tc_values, [0] * len(tc_values), c=alloy_dataset['Cluster'], cmap='viridis')
plt.scatter(kmeans.cluster_centers_, [0] * len(kmeans.cluster_centers_), c='red', marker='x')
plt.xlabel('Critical Temperature (Tc)')
plt.title('Clustering of Alloys Based on Tc')
plt.show()
"""

jitter = np.random.normal(0, 0.1, len(tc_values))  # Adding a small random jitter
plt.scatter(tc_values, jitter, c=alloy_dataset['Cluster'], cmap='viridis')
plt.xlabel('Critical Temperature (Tc)')
plt.title('Clustering of Alloys Based on Tc with Jitter')
plt.show()


features = alloy_dataset[['critical_temp', 'mean_ThermalConductivity']]  # Example of two features
#kmeans = KMeans(n_clusters=5, random_state=42)
alloy_dataset['Cluster'] = kmeans.fit_predict(features)

plt.scatter(alloy_dataset['critical_temp'], alloy_dataset['mean_ThermalConductivity'], c=alloy_dataset['Cluster'], cmap='viridis')
plt.xlabel('Critical Temperature (Tc)')
plt.ylabel('Atomic Mass')
plt.title('Clustering of Alloys Based on Tc and Atomic Mass')
plt.show()