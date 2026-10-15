import os
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. SETUP
# ============================================================

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# 2. LOAD EXISTING RESULTS
# ============================================================

kmeans = pd.read_csv(
    f"{RESULTS_DIR}/kmeans_metrics.csv"
)

dbscan = pd.read_csv(
    f"{RESULTS_DIR}/dbscan_metrics.csv"
)

dec = pd.read_csv(
    f"{RESULTS_DIR}/dec_metrics.csv"
)

cluster_analysis = pd.read_csv(
    f"{RESULTS_DIR}/dec_cluster_analysis.csv"
)

comparison = pd.read_csv(
    f"{RESULTS_DIR}/model_comparison.csv"
)

dec_users = pd.read_csv(
    f"{RESULTS_DIR}/dec_user_groups.csv"
)


# ============================================================
# 3. DATASET INFORMATION
# ============================================================

dataset_path = "data/disaster_users.csv"

dataset = pd.read_csv(dataset_path)

number_of_users = len(dataset)

original_features = [
    "x_coordinate",
    "y_coordinate",
    "channel_condition",
    "emergency_priority",
    "requested_data_type",
    "delay_requirement",
    "semantic_similarity"
]

processed_data = pd.read_csv(
    "data/processed_users.csv"
)

processed_feature_count = processed_data.shape[1]


# ============================================================
# 4. PRINT FINAL REPORT
# ============================================================

print("\n")
print("=" * 75)
print("                 FINAL PROJECT RESULTS")
print("=" * 75)

print("\nPROJECT TITLE:")
print(
    "Deep Embedded Clustering-Based Semantic User "
    "Grouping for UAV-Assisted Disaster Communication"
)

print("\n" + "-" * 75)
print("DATASET INFORMATION")
print("-" * 75)

print("Number of users:", number_of_users)
print("Original feature columns:", len(original_features))
print("Processed feature dimensions:", processed_feature_count)


# ============================================================
# 5. MODEL COMPARISON
# ============================================================

print("\n" + "-" * 75)
print("MODEL COMPARISON")
print("-" * 75)

display_columns = [
    "Method",
    "Clusters",
    "Silhouette_Score",
    "Davies_Bouldin_Index",
    "Calinski_Harabasz_Score"
]

print(
    comparison[display_columns].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 6. BEST MODEL BY EACH METRIC
# ============================================================

best_silhouette_row = comparison.loc[
    comparison["Silhouette_Score"].idxmax()
]

best_db_row = comparison.loc[
    comparison["Davies_Bouldin_Index"].idxmin()
]

best_ch_row = comparison.loc[
    comparison["Calinski_Harabasz_Score"].idxmax()
]

print("\n" + "-" * 75)
print("BEST METHOD BY METRIC")
print("-" * 75)

print(
    "Highest Silhouette Score:",
    best_silhouette_row["Method"],
    f"({best_silhouette_row['Silhouette_Score']:.4f})"
)

print(
    "Lowest Davies-Bouldin Index:",
    best_db_row["Method"],
    f"({best_db_row['Davies_Bouldin_Index']:.4f})"
)

print(
    "Highest Calinski-Harabasz Score:",
    best_ch_row["Method"],
    f"({best_ch_row['Calinski_Harabasz_Score']:.4f})"
)


# ============================================================
# 7. DEC CLUSTER ANALYSIS
# ============================================================

print("\n" + "-" * 75)
print("DEC SEMANTIC USER GROUP ANALYSIS")
print("-" * 75)

print(
    cluster_analysis.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}"
    )
)


# ============================================================
# 8. DETERMINE CLUSTER CHARACTERISTICS
# ============================================================

highest_priority_cluster = cluster_analysis.loc[
    cluster_analysis["Avg_Priority"].idxmax()
]

lowest_delay_cluster = cluster_analysis.loc[
    cluster_analysis["Avg_Delay"].idxmin()
]

highest_similarity_cluster = cluster_analysis.loc[
    cluster_analysis["Avg_Semantic_Similarity"].idxmax()
]

print("\n" + "-" * 75)
print("IMPORTANT DEC CLUSTER CHARACTERISTICS")
print("-" * 75)

print(
    "Highest priority cluster:",
    int(highest_priority_cluster["dec_cluster"]),
    f"(Average priority: "
    f"{highest_priority_cluster['Avg_Priority']:.3f})"
)

print(
    "Lowest delay cluster:",
    int(lowest_delay_cluster["dec_cluster"]),
    f"(Average delay: "
    f"{lowest_delay_cluster['Avg_Delay']:.3f})"
)

print(
    "Highest semantic similarity cluster:",
    int(highest_similarity_cluster["dec_cluster"]),
    f"(Average similarity: "
    f"{highest_similarity_cluster['Avg_Semantic_Similarity']:.3f})"
)


# ============================================================
# 9. MOST COMMON DATA TYPE FOR EACH CLUSTER
# ============================================================

print("\n" + "-" * 75)
print("MOST COMMON DATA TYPE BY DEC CLUSTER")
print("-" * 75)

cluster_data_types = {}

for cluster in sorted(dec_users["dec_cluster"].unique()):

    cluster_users = dec_users[
        dec_users["dec_cluster"] == cluster
    ]

    most_common_type = (
        cluster_users["requested_data_type"]
        .value_counts()
        .idxmax()
    )

    cluster_data_types[int(cluster)] = most_common_type

    print(
        f"Cluster {int(cluster)}: "
        f"{most_common_type}"
    )


# ============================================================
# 10. CREATE FINAL CSV
# ============================================================

final_results = comparison.copy()

final_results.to_csv(
    f"{RESULTS_DIR}/final_project_results.csv",
    index=False
)


# ============================================================
# 11. CREATE TEXT REPORT
# ============================================================

report_lines = []

report_lines.append(
    "FINAL PROJECT REPORT\n"
)

report_lines.append(
    "Project Title: Deep Embedded Clustering-Based Semantic "
    "User Grouping for UAV-Assisted Disaster Communication\n"
)

report_lines.append(
    f"\nNumber of users: {number_of_users}"
)

report_lines.append(
    f"\nProcessed feature dimensions: {processed_feature_count}"
)

report_lines.append(
    "\n\nMODEL COMPARISON\n"
)

report_lines.append(
    comparison[display_columns].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

report_lines.append(
    "\n\nBEST METHODS BY METRIC\n"
)

report_lines.append(
    f"Highest Silhouette Score: "
    f"{best_silhouette_row['Method']} "
    f"({best_silhouette_row['Silhouette_Score']:.4f})"
)

report_lines.append(
    f"\nLowest Davies-Bouldin Index: "
    f"{best_db_row['Method']} "
    f"({best_db_row['Davies_Bouldin_Index']:.4f})"
)

report_lines.append(
    f"\nHighest Calinski-Harabasz Score: "
    f"{best_ch_row['Method']} "
    f"({best_ch_row['Calinski_Harabasz_Score']:.4f})"
)

report_lines.append(
    "\n\nDEC CLUSTER CHARACTERISTICS\n"
)

report_lines.append(
    cluster_analysis.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}"
    )
)

report_lines.append(
    "\n\nMOST COMMON DATA TYPE BY CLUSTER\n"
)

for cluster, data_type in cluster_data_types.items():

    report_lines.append(
        f"Cluster {cluster}: {data_type}"
    )

report_lines.append("\n\nINTERPRETATION AND LIMITATIONS\n")
report_lines.append("The table above contains measured clustering metrics from the result CSV files. Similar values indicate no meaningful measured advantage for one method on these metrics. Cluster names summarize observed cluster characteristics. The groups may inform downstream UAV communication research; this project does not perform flight control, packet routing, or resource allocation.\n")


# ============================================================
# 12. WRITE TEXT REPORT
# ============================================================

with open(
    f"{RESULTS_DIR}/final_project_report.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "\n".join(report_lines)
    )


# ============================================================
# 13. FINAL MODEL COMPARISON VISUALIZATION
# ============================================================

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

axes[0].set_title(
    "Silhouette Score"
)

axes[0].set_ylabel(
    "Score"
)

axes[0].set_ylim(
    0,
    1
)


# Davies-Bouldin Index
axes[1].bar(
    methods,
    comparison["Davies_Bouldin_Index"]
)

axes[1].set_title(
    "Davies-Bouldin Index"
)

axes[1].set_ylabel(
    "Score"
)


# Calinski-Harabasz Score
axes[2].bar(
    methods,
    comparison["Calinski_Harabasz_Score"]
)

axes[2].set_title(
    "Calinski-Harabasz Score"
)

axes[2].set_ylabel(
    "Score"
)


plt.tight_layout()

plt.savefig(
    f"{RESULTS_DIR}/final_model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 14. COMPLETION MESSAGE
# ============================================================

print("\n" + "=" * 75)
print("FINAL PROJECT REPORT CREATED SUCCESSFULLY!")
print("=" * 75)

print("\nGenerated files:")

print(
    "results/final_project_results.csv"
)

print(
    "results/final_project_report.txt"
)

print(
    "results/final_model_comparison.png"
)

print("\nAll existing ML results were preserved.")
