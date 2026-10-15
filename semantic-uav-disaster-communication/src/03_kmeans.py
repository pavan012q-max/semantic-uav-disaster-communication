import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

# Load preprocessed data
X = pd.read_csv("data/processed_users.csv").values

print("Processed data shape:", X.shape)

# --------------------------------------------------
# Find a suitable number of clusters
# --------------------------------------------------

k_values = range(2, 9)
silhouette_scores = []

for k in k_values:
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20
    )

    labels = model.fit_predict(X)

    score = silhouette_score(X, labels)
    silhouette_scores.append(score)

    print(f"K = {k}, Silhouette Score = {score:.4f}")

# Select k with highest Silhouette Score
best_k = list(k_values)[silhouette_scores.index(max(silhouette_scores))]

print("\nBest number of clusters:", best_k)

# --------------------------------------------------
# Final K-Means model
# --------------------------------------------------

kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=20
)

labels = kmeans.fit_predict(X)

# --------------------------------------------------
# Evaluation metrics
# --------------------------------------------------

silhouette = silhouette_score(X, labels)
davies_bouldin = davies_bouldin_score(X, labels)
calinski_harabasz = calinski_harabasz_score(X, labels)

print("\nK-Means Results")
print("----------------------------")
print(f"Number of clusters: {best_k}")
print(f"Silhouette Score: {silhouette:.4f}")
print(f"Davies-Bouldin Index: {davies_bouldin:.4f}")
print(f"Calinski-Harabasz Score: {calinski_harabasz:.4f}")

# --------------------------------------------------
# Save cluster assignments
# --------------------------------------------------

original_df = pd.read_csv("data/disaster_users.csv")

original_df["kmeans_cluster"] = labels

original_df.to_csv(
    "results/kmeans_user_groups.csv",
    index=False
)

# Save metrics
metrics = pd.DataFrame({
    "Method": ["K-Means"],
    "Clusters": [best_k],
    "Silhouette_Score": [silhouette],
    "Davies_Bouldin_Index": [davies_bouldin],
    "Calinski_Harabasz_Score": [calinski_harabasz]
})

metrics.to_csv(
    "results/kmeans_metrics.csv",
    index=False
)

# --------------------------------------------------
# Visualization
# --------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    original_df["x_coordinate"],
    original_df["y_coordinate"],
    c=labels,
    s=25,
    alpha=0.7
)

plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.title("K-Means User Grouping")
plt.grid(alpha=0.2)

plt.savefig(
    "results/kmeans_clusters.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nK-Means results saved successfully!")
