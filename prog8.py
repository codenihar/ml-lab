import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Step 1: Generate and Save a Random Dataset
np.random.seed(42)
data = {
    "Feature1": np.random.rand(100) * 100,  # Random numbers between 0 and 100
    "Feature2": np.random.rand(100) * 100
}
df = pd.DataFrame(data)
df.to_csv("kmeans_data.csv", index=False)
print("Data saved to 'kmeans_data.csv'.")

# Step 2: Load the Dataset
data = pd.read_csv("kmeans_data.csv")

# Step 3: Apply K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(data)

# Step 4: Visualize the Clusters
plt.figure(figsize=(8, 6))
colors = ['red', 'blue', 'green']
for cluster in range(3):
    cluster_data = data[data['Cluster'] == cluster]
    plt.scatter(cluster_data['Feature1'], cluster_data['Feature2'], 
                color=colors[cluster], label=f'Cluster {cluster}')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            color='yellow', marker='X', s=200, label='Centroids')

plt.title("K-Means Clustering")
plt.xlabel("Feature1")
plt.ylabel("Feature2")
plt.legend()
plt.grid()
plt.show()
