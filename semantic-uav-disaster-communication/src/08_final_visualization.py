import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("results", exist_ok=True)

# Load DEC cluster analysis
df = pd.read_csv("results/dec_cluster_analysis.csv")

# Cluster labels based on the actual analysis
cluster_names = {
    0: "General",
    1: "Rescue",
    2: "Location",
    3: "Medical"
}

df["Cluster_Name"] = df["dec_cluster"].map(cluster_names)

# --------------------------------------------------
# 1. Number of users in each cluster
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.bar(
    df["Cluster_Name"],
    df["Users"]
)

plt.title("DEC Semantic User Groups")
plt.xlabel("User Group")
plt.ylabel("Number of Users")

plt.tight_layout()

plt.savefig(
    "results/dec_user_group_sizes.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# 2. Emergency priority comparison
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.bar(
    df["Cluster_Name"],
    df["Avg_Priority"]
)

plt.title("Average Emergency Priority by DEC Cluster")
plt.xlabel("User Group")
plt.ylabel("Average Emergency Priority")

plt.tight_layout()

plt.savefig(
    "results/dec_priority_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# 3. Delay requirement comparison
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.bar(
    df["Cluster_Name"],
    df["Avg_Delay"]
)

plt.title("Average Delay Requirement by DEC Cluster")
plt.xlabel("User Group")
plt.ylabel("Average Delay Requirement")

plt.tight_layout()

plt.savefig(
    "results/dec_delay_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nFinal visualizations created successfully!")

print("\nFiles saved:")
print("results/dec_user_group_sizes.png")
print("results/dec_priority_comparison.png")
print("results/dec_delay_comparison.png")
