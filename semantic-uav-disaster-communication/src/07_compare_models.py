import os
import pandas as pd
import matplotlib.pyplot as plt

# Create results folder if it doesn't exist
os.makedirs("results", exist_ok=True)

# --------------------------------------------------
# Load results from all three models
# --------------------------------------------------

kmeans = pd.read_csv("results/kmeans_metrics.csv")
dbscan = pd.read_csv("results/dbscan_metrics.csv")
dec = pd.read_csv("results/dec_metrics.csv")

# Combine results
comparison = pd.concat(
    [kmeans, dbscan, dec],
    ignore_index=True
)

# Keep only the important comparison columns
comparison = comparison[
    [
        "Method",
        "Clusters",
        "Silhouette_Score",
        "Davies_Bouldin_Index",
        "Calinski_Harabasz_Score"
    ]
]

# --------------------------------------------------
# Display comparison
# --------------------------------------------------

print("\nMODEL COMPARISON")
print("=" * 80)

print(
    comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

# --------------------------------------------------
# Determine best method for each metric
# --------------------------------------------------

best_silhouette = comparison.loc[
    comparison["Silhouette_Score"].idxmax(),
    "Method"
]

best_db = comparison.loc[
    comparison["Davies_Bouldin_Index"].idxmin(),
    "Method"
]

best_ch = comparison.loc[
    comparison["Calinski_Harabasz_Score"].idxmax(),
    "Method"
]

print("\nBest Method by Metric")
print("-" * 40)

print("Best Silhouette Score:", best_silhouette)
print("Best Davies-Bouldin Index:", best_db)
print("Best Calinski-Harabasz Score:", best_ch)

# --------------------------------------------------
# Save comparison table
# --------------------------------------------------

comparison.to_csv(
    "results/model_comparison.csv",
    index=False
)

# --------------------------------------------------
# Create comparison graph
# --------------------------------------------------

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 5)
)

methods = comparison["Method"]

# Silhouette Score
axes[0].bar(
    methods,
    comparison["Silhouette_Score"]
)

axes[0].set_title("Silhouette Score")
axes[0].set_ylabel("Score")
axes[0].set_ylim(0, 1)

# Davies-Bouldin Index
axes[1].bar(
    methods,
    comparison["Davies_Bouldin_Index"]
)

axes[1].set_title("Davies-Bouldin Index")
axes[1].set_ylabel("Score")

# Calinski-Harabasz Score
axes[2].bar(
    methods,
    comparison["Calinski_Harabasz_Score"]
)

axes[2].set_title("Calinski-Harabasz Score")
axes[2].set_ylabel("Score")

plt.tight_layout()

plt.savefig(
    "results/model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nComparison saved successfully!")
print("results/model_comparison.csv")
print("results/model_comparison.png")
