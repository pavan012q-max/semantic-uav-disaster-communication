"""Streamlit dashboard built from the project's generated CSV and image files."""
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
TITLE = "Deep Embedded Clustering-Based Semantic User Grouping for UAV-Assisted Disaster Communication"

st.set_page_config(page_title="Semantic UAV User Grouping", layout="wide")
st.title(TITLE)
st.markdown("### Problem statement")
st.write("Disaster communication systems must serve users with different locations, channel conditions, urgency, data needs, delay limits, and semantic similarity. This project groups users from those factors to provide an input for later communication-management systems.")
st.markdown("### Input features")
st.write("Location (x/y), channel condition, emergency priority, requested data type, delay requirement, and semantic similarity.")

dataset = pd.read_csv(DATA / "disaster_users.csv")
processed = pd.read_csv(DATA / "processed_users.csv")
c1, c2, c3 = st.columns(3)
c1.metric("Users", len(dataset)); c2.metric("Raw columns", dataset.shape[1]); c3.metric("Processed features", processed.shape[1])
st.markdown("### Preprocessing")
st.write("Six numerical features are standardized. Requested data type is one-hot encoded. The user_id column is an identifier and excluded. One-hot encoding adds four category columns, so six numeric plus four encoded features produce 10 features.")

st.markdown("### Model results and comparison")
for fname in ("kmeans_metrics.csv", "dbscan_metrics.csv", "dec_metrics.csv"):
    path = RESULTS / fname
    if path.exists():
        st.subheader(fname.replace("_metrics.csv", "").replace("_", " ").title())
        st.dataframe(pd.read_csv(path), use_container_width=True)
comparison = RESULTS / "model_comparison.csv"
if comparison.exists():
    st.subheader("Model comparison")
    st.dataframe(pd.read_csv(comparison), use_container_width=True)
    st.caption("Metrics are measured results. Similar scores do not support a claim that DEC significantly outperforms the baselines.")

analysis_path = RESULTS / "dec_cluster_analysis.csv"
if analysis_path.exists():
    analysis = pd.read_csv(analysis_path)
    st.markdown("### DEC cluster sizes and characteristics")
    st.dataframe(analysis, use_container_width=True)
    if "Group_Name" in analysis:
        for _, row in analysis.iterrows():
            st.write(f"Cluster {int(row['dec_cluster'])}: **{row['Group_Name']}** — most common data type: {row['Most_Common_Data_Type']}; average priority {row['Avg_Priority']:.2f}, delay requirement {row['Avg_Delay']:.2f}, semantic similarity {row['Avg_Semantic_Similarity']:.2f}.")

st.markdown("### Visualizations")
for name in ("kmeans_clusters.png", "dbscan_clusters.png", "dec_clusters.png", "model_comparison.png", "dec_user_group_sizes.png", "dec_priority_comparison.png", "dec_delay_comparison.png", "final_model_comparison.png"):
    path = RESULTS / name
    if path.exists():
        st.image(str(path), caption=name.replace(".png", "").replace("_", " ").title(), use_container_width=True)

st.markdown("### Complete pipeline")
st.write("Inspect data → preprocess and save the fitted transformation → K-Means baseline → DBSCAN baseline → autoencoder pretraining and DEC clustering → analyze clusters → compare metrics → create visualizations → generate the final report.")
st.markdown("### Potential use in UAV-assisted disaster communication")
st.write("Semantic groups can be supplied to future systems as context for prioritizing communication or studying service strategies. This project performs grouping only; it does not control UAV flight, route packets, or allocate communication resources.")
