# Deep Embedded Clustering-Based Semantic User Grouping for UAV-Assisted Disaster Communication

## Abstract

This college project applies K-Means, DBSCAN, and Deep Embedded Clustering (DEC) to a synthetic dataset of 1,000 disaster-area users. It groups users with similar location, communication conditions, urgency, requested data, delay needs, and semantic similarity. The output is semantic user groups that could inform later UAV communication research. The project does not control UAVs, route packets, or allocate network resources.

## Problem statement and objective

Users in a disaster have different communication needs. Treating all users alike can hide urgent or semantically related requests. The objective is to preprocess user attributes, learn groups, compare three clustering methods, describe the groups, and demonstrate deployment-style assignment of one new user using the saved DEC model.

## Dataset

`data/disaster_users.csv` contains 1,000 synthetic users and seven feature columns, plus the `user_id` identifier (eight columns total). `user_id` identifies a record and is not a clustering feature.

The six conceptual factors are:

1. **Location:** `x_coordinate` and `y_coordinate` (two raw columns)
2. **Channel condition**
3. **Emergency priority**
4. **Requested data type**
5. **Delay requirement**
6. **Semantic similarity**

There are seven feature columns because location is represented by two coordinates and the other five factors each use one column; `user_id` is an additional identifier column. Preprocessing excludes the identifier, standardizes the six numeric columns, and one-hot encodes the four observed data types. This produces six numeric plus four categorical indicator features, or 10 processed features in `data/processed_users.csv`. The fitted transformer is saved as `results/preprocessor.joblib`; unseen data types are safely represented by all-zero category indicators.

## System architecture

`01_inspect_data.py` → `02_preprocess.py` → `03_kmeans.py` → `04_dbscan.py` → `05_dec.py` → `06_cluster_analysis.py` → `07_compare_models.py` → `08_final_visualization.py` → `09_final_report.py`.

The autoencoder learns a compact latent representation by encoding the 10-dimensional inputs and reconstructing them. DEC initializes cluster centers with K-Means in latent space, then refines the encoder and centers using a sharpened target distribution and Student-t soft assignments. New-user inference loads the fitted preprocessor, saved encoder, and trained cluster centers; it does not retrain.

K-Means partitions points around a chosen number of centroids. DBSCAN forms density-connected groups and can mark outliers as noise. DEC jointly learns a neural embedding and clusters it. They provide complementary baselines; the reported metrics should be interpreted on the evaluated feature representation and together with group characteristics.

## Evaluation metrics

- **Silhouette:** higher is better; compares within-cluster cohesion with separation.
- **Davies–Bouldin:** lower is better; summarizes cluster similarity.
- **Calinski–Harabasz:** higher is better; compares between-cluster dispersion with within-cluster dispersion.

These internal metrics do not establish real-world communication performance.

## Current results

The pipeline writes current measured values to `results/kmeans_metrics.csv`, `results/dbscan_metrics.csv`, `results/dec_metrics.csv`, and `results/model_comparison.csv`. Use those generated CSVs as the source of truth; the final report and dashboard load them. Existing checked-in outputs show approximately four clusters and near-identical baseline/DEC scores (Silhouette 0.6320, Davies–Bouldin 0.5501, Calinski–Harabasz 2630.6143). Small run-to-run/library differences may occur, although seeds are set for reproducibility. Do not claim DEC significantly outperforms the baselines on these results.

Cluster IDs are arbitrary. `results/dec_cluster_analysis.csv` includes each cluster's most common requested data type and its interpreted group name (General, Rescue, Location, or Medical), alongside measured averages. The meanings summarize the current data and may change if training or data changes.

## Install and run

Use Python 3.10–3.12 in a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run each pipeline stage with `python src/01_inspect_data.py` through `python src/09_final_report.py`, or run the convenience command:

```powershell
python run_all.py
```

`run_all.py` stops at the first failed stage, then starts the dashboard. Press Ctrl+C to stop the dashboard. To open the dashboard separately:

```powershell
python -m streamlit run src/11_dashboard.py
```

## New-user prediction

Run the pipeline first to create deployment artifacts. Then provide all seven user attributes; for example:

```powershell
python src/10_predict_new_user.py --x_coordinate 30 --y_coordinate 20 --channel_condition 0.75 --emergency_priority 3 --requested_data_type Medical --delay_requirement 0.8 --semantic_similarity 0.9
```

The script prints the input values, predicted DEC cluster, interpreted group, assignment confidence, and an explanation based on that cluster's measured characteristics. It saves the row to `results/new_user_prediction.csv`. An unknown `requested_data_type` is accepted by the saved one-hot encoder's unknown-category handling.

## Outputs

CSV metrics, per-user cluster assignments, cluster analysis, comparison tables, prediction output, plots, saved DEC inference artifacts, and the final report are stored under `results/`. The dashboard reads these output files rather than embedding experiment metrics in code.

## Limitations and future scope

The dataset is synthetic, and internal clustering scores do not prove that groups improve disaster response. Results depend on feature scales, chosen cluster count, model settings, and random seeds. The current project only groups users. Future work could validate groups on real and ethically collected data, test downstream communication simulations, evaluate fairness and robustness, and connect the groups to separately designed UAV communication or resource-management systems.
