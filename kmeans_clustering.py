# kmeans_clustering.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Step 1: Load dataset
df = pd.read_csv("Mall_Customers.csv")
print("✅ Data Loaded Successfully")
print(df.head())

# Step 2: Select features
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Step 3: Scale data (optional but helps KMeans)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Elbow method to find optimal K
inertia = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.plot(K_range, inertia, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("Inertia")
plt.grid(True)
plt.savefig("elbow_plot.png")
plt.show(block=True)

# Step 5: Choose K = 5 and fit
kmeans = KMeans(n_clusters=5, random_state=42)
labels = kmeans.fit_predict(X_scaled)

# Add cluster labels to original dataframe
df['Cluster'] = labels

# Step 6: Visualize clusters
plt.figure(figsize=(8,6))
for i in range(5):
    plt.scatter(X_scaled[labels == i, 0], X_scaled[labels == i, 1], label=f'Cluster {i}')
plt.title("Customer Segments")
plt.xlabel("Annual Income (scaled)")
plt.ylabel("Spending Score (scaled)")
plt.legend()
plt.savefig("clusters.png")
plt.show(block=True)

# Step 7: Evaluate using Silhouette Score
score = silhouette_score(X_scaled, labels)
print(f"🔍 Silhouette Score: {score:.2f}")
