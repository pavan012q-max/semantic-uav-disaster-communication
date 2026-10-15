import pandas as pd

# Load the original dataset
df = pd.read_csv("data/disaster_users.csv")

# Load DEC cluster assignments
dec_groups = pd.read_csv("results/dec_user_groups.csv")

# Add DEC cluster labels
df["dec_cluster"] = dec_groups["dec_cluster"]

print("\nDEC CLUSTER ANALYSIS")
print("=" * 60)

# Overall cluster counts
print("\nNumber of users in each cluster:")
print(df["dec_cluster"].value_counts().sort_index())

# Analyze numerical characteristics
summary = df.groupby("dec_cluster").agg(
    Users=("user_id", "count"),
    Avg_X=("x_coordinate", "mean"),
    Avg_Y=("y_coordinate", "mean"),
    Avg_Channel=("channel_condition", "mean"),
    Avg_Priority=("emergency_priority", "mean"),
    Avg_Delay=("delay_requirement", "mean"),
    Avg_Semantic_Similarity=("semantic_similarity", "mean")
)
summary["Most_Common_Data_Type"] = df.groupby("dec_cluster")["requested_data_type"].agg(
    lambda values: values.value_counts().idxmax()
)
summary["Group_Name"] = summary["Most_Common_Data_Type"].map({
    "General": "General", "Rescue": "Rescue",
    "Location": "Location", "Medical": "Medical"
})

print("\nCluster Characteristics:")
print(summary.round(3).to_string())

# Most common requested data type in each cluster
print("\nMost Common Data Type in Each Cluster:")

for cluster in sorted(df["dec_cluster"].unique()):

    cluster_data = df[df["dec_cluster"] == cluster]

    most_common = (
        cluster_data["requested_data_type"]
        .value_counts()
        .idxmax()
    )

    print(
        f"Cluster {cluster}: "
        f"{most_common}"
    )

# Save analysis
summary.to_csv(
    "results/dec_cluster_analysis.csv"
)

print("\nAnalysis saved to:")
print("results/dec_cluster_analysis.csv")
