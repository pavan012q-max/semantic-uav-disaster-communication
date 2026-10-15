import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

# Load processed data
X = pd.read_csv("data/processed_users.csv").values

print("Processed data shape:", X.shape)

# Create results folder if it does not exist
os.makedirs("results", exist_ok=True)

# --------------------------------------------------
# Try different DBSCAN parameter combinations
# --------------------------------------------------

eps_values = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
min_samples_values = [5, 10, 15]

best_score = -1
best_eps = None
best_min_samples = None
best_labels = None

print("\nDBSCAN Parameter Search")
print("----------------------------")

for eps in eps_values:
    for min_samples in min_samples_values:

        model = DBSCAN(
            eps=eps,
            min_samples=min_samples
        )

        labels = model.fit_predict(X)

        # Ignore noise points (-1) for evaluation
        mask = labels != -1

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = sum(labels == -1)

        # Need at least 2 clusters for these metrics
        if n_clusters >= 2 and mask.sum() > n_clusters:

            score = silhouette_score(
                X[mask],
                labels[mask]
            )

            print(
                f"eps={eps}, min_samples={min_samples}, "
                f"clusters={n_clusters}, noise={n_noise}, "
                f"silhouette={score:.4f}"
            )

            if score > best_score:
                best_score = score
                best_eps = eps
                best_min_samples = min_samples
                best_labels = labels


# --------------------------------------------------
# Check best model
# --------------------------------------------------

print("\nBest DBSCAN Parameters")
print("----------------------------")
print("Best eps:", best_eps)
print("Best min_samples:", best_min_samples)

labels = best_labels

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = sum(labels == -1)

print("Number of clusters:", n_clusters)
print("Number of noise points:", n_noise)

# --------------------------------------------------
# Evaluation
# --------------------------------------------------

mask = labels != -1

silhouette = silhouette_score(
    X[mask],
    labels[mask]
)

davies_bouldin = davies_bouldin_score(
    X[mask],
    labels[mask]
)

calinski_harabasz = calinski_harabasz_score(
    X[mask],
    labels[mask]
)

print("\nDBSCAN Results")
print("----------------------------")
print(f"Silhouette Score: {silhouette:.4f}")
print(f"Davies-Bouldin Index: {davies_bouldin:.4f}")
print(f"Calinski-Harabasz Score: {calinski_harabasz:.4f}")
print(f"Noise Points: {n_noise}")

# --------------------------------------------------
# Save user groups
# --------------------------------------------------

original_df = pd.read_csv("data/disaster_users.csv")

original_df["dbscan_cluster"] = labels

original_df.to_csv(
    "results/dbscan_user_groups.csv",
    index=False
)

# --------------------------------------------------
# Save metrics
# --------------------------------------------------

metrics = pd.DataFrame({
    "Method": ["DBSCAN"],
    "Clusters": [n_clusters],
    "Silhouette_Score": [silhouette],
    "Davies_Bouldin_Index": [davies_bouldin],
    "Calinski_Harabasz_Score": [calinski_harabasz],
    "Noise_Points": [n_noise]
})

metrics.to_csv(
    "results/dbscan_metrics.csv",
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
plt.title("DBSCAN User Grouping")
plt.grid(alpha=0.2)

plt.savefig(
    "results/dbscan_clusters.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nDBSCAN results saved successfully!")
