import os
import itertools
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

import tensorflow as tf
from keras import Model, layers, optimizers

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except (AttributeError, RuntimeError):
    pass


# ============================================================
# 1. LOAD DATA
# ============================================================

X = pd.read_csv("data/processed_users.csv").values.astype("float32")

print("Processed data shape:", X.shape)


# ============================================================
# 2. AUTOENCODER
# ============================================================

input_dim = X.shape[1]
latent_dim = 5

inputs = layers.Input(shape=(input_dim,))

# Encoder
encoded = layers.Dense(32, activation="relu")(inputs)
encoded = layers.Dense(16, activation="relu")(encoded)
latent = layers.Dense(latent_dim, activation="linear", name="latent")(encoded)

# Decoder
decoded = layers.Dense(16, activation="relu")(latent)
decoded = layers.Dense(32, activation="relu")(decoded)
outputs = layers.Dense(input_dim, activation="linear")(decoded)

autoencoder = Model(inputs, outputs)
encoder = Model(inputs, latent)

autoencoder.compile(
    optimizer="adam",
    loss="mse"
)

print("\nTraining autoencoder...")

autoencoder.fit(
    X,
    X,
    epochs=100,
    batch_size=64,
    shuffle=True,
    verbose=0
)

print("Autoencoder training completed!")


# ============================================================
# 3. GET DEEP EMBEDDINGS
# ============================================================

Z = encoder.predict(X, verbose=0)

print("Latent representation shape:", Z.shape)


# ============================================================
# 4. INITIALIZE CLUSTERS USING K-MEANS
# ============================================================

n_clusters = 4

kmeans = KMeans(
    n_clusters=n_clusters,
    random_state=42,
    n_init=20
)

initial_labels = kmeans.fit_predict(Z)

cluster_centers = kmeans.cluster_centers_.astype("float32")


# ============================================================
# 5. DEC CLUSTERING LAYER
# ============================================================

class ClusteringLayer(layers.Layer):

    def __init__(self, n_clusters, alpha=1.0, **kwargs):
        super().__init__(**kwargs)

        self.n_clusters = n_clusters
        self.alpha = alpha

    def build(self, input_shape):

        self.clusters = self.add_weight(
            shape=(self.n_clusters, input_shape[-1]),
            initializer="zeros",
            trainable=True,
            name="cluster_centers"
        )

    def call(self, inputs):

        distance = tf.reduce_sum(
            tf.square(
                tf.expand_dims(inputs, axis=1)
                - self.clusters
            ),
            axis=2
        )

        q = 1.0 / (
            1.0 + distance / self.alpha
        )

        q = tf.pow(q, (self.alpha + 1.0) / 2.0)

        q = q / tf.reduce_sum(
            q,
            axis=1,
            keepdims=True
        )

        return q


# ============================================================
# 6. BUILD DEC MODEL
# ============================================================

dec_input = layers.Input(shape=(input_dim,))

dec_latent = encoder(dec_input)

clustering_layer = ClusteringLayer(
    n_clusters=n_clusters,
    name="clustering"
)

q = clustering_layer(dec_latent)

dec_model = Model(
    dec_input,
    q
)

# Initialize cluster centers from K-Means
clustering_layer.set_weights(
    [cluster_centers]
)

print("\nDEC model initialized with K-Means centers.")


# ============================================================
# 7. TARGET DISTRIBUTION
# ============================================================

def target_distribution(q):

    weight = q ** 2 / np.sum(q, axis=0)

    return (
        weight.T /
        np.sum(weight, axis=1)
    ).T


# ============================================================
# 8. DEC TRAINING
# ============================================================

optimizer = optimizers.Adam(
    learning_rate=0.001
)

max_iterations = 100
update_interval = 5

print("\nTraining DEC...")

for iteration in range(max_iterations):

    # Current soft assignments
    q_values = dec_model.predict(
        X,
        verbose=0
    )

    # Update target distribution
    if iteration % update_interval == 0:

        p_values = target_distribution(
            q_values
        )

    # Train using KL divergence
    with tf.GradientTape() as tape:

        q_tensor = dec_model(X, training=True)

        p_tensor = tf.convert_to_tensor(
            p_values,
            dtype=tf.float32
        )

        loss = tf.reduce_mean(
            tf.reduce_sum(
                p_tensor *
                tf.math.log(
                    (p_tensor + 1e-8) /
                    (q_tensor + 1e-8)
                ),
                axis=1
            )
        )

    gradients = tape.gradient(
        loss,
        dec_model.trainable_variables
    )

    optimizer.apply_gradients(
        zip(
            gradients,
            dec_model.trainable_variables
        )
    )

    if iteration % 10 == 0:

        print(
            f"Iteration {iteration:3d} | "
            f"KL Loss: {loss.numpy():.6f}"
        )


print("DEC training completed!")


# ============================================================
# 9. FINAL CLUSTER ASSIGNMENTS
# ============================================================

q_final = dec_model.predict(
    X,
    verbose=0
)

Z_final = encoder.predict(
    X,
    verbose=0
)

dec_labels = np.argmax(
    q_final,
    axis=1
)


# ============================================================
# 10. EVALUATION
# ============================================================

silhouette = silhouette_score(
    X,
    dec_labels
)

davies_bouldin = davies_bouldin_score(
    X,
    dec_labels
)

calinski_harabasz = calinski_harabasz_score(
    X,
    dec_labels
)

print("\nDEC Results")
print("----------------------------")
print("Number of clusters:", n_clusters)
print(f"Silhouette Score: {silhouette:.4f}")
print(f"Davies-Bouldin Index: {davies_bouldin:.4f}")
print(f"Calinski-Harabasz Score: {calinski_harabasz:.4f}")


# ============================================================
# 11. LABEL VALIDATION AGAINST BASELINES
# ============================================================

def permutation_invariant_agreement(reference, candidate):

    reference_labels = sorted(np.unique(reference))
    candidate_labels = sorted(np.unique(candidate))

    best_agreement = 0.0

    for permutation in itertools.permutations(
        reference_labels,
        len(candidate_labels)
    ):

        mapping = dict(zip(candidate_labels, permutation))
        remapped = np.array([
            mapping[label]
            for label in candidate
        ])

        best_agreement = max(
            best_agreement,
            np.mean(reference == remapped)
        )

    return best_agreement


kmeans_labels = pd.read_csv(
    "results/kmeans_user_groups.csv"
)["kmeans_cluster"].to_numpy()

dbscan_labels = pd.read_csv(
    "results/dbscan_user_groups.csv"
)["dbscan_cluster"].to_numpy()

kmeans_agreement = permutation_invariant_agreement(
    kmeans_labels,
    dec_labels
)

dbscan_agreement = permutation_invariant_agreement(
    dbscan_labels,
    dec_labels
)

print(
    f"K-Means vs DEC label agreement (permuted): "
    f"{kmeans_agreement:.4%}"
)
print(
    f"DBSCAN vs DEC label agreement (permuted): "
    f"{dbscan_agreement:.4%}"
)


# ============================================================
# 12. SAVE RESULTS
# ============================================================

os.makedirs(
    "results",
    exist_ok=True
)

original_df = pd.read_csv(
    "data/disaster_users.csv"
)

original_df["dec_cluster"] = dec_labels

original_df.to_csv(
    "results/dec_user_groups.csv",
    index=False
)

# Persist the exact trained encoder, DEC centers, and cluster interpretation
# metadata so prediction never retrains on a new user.
np.savez_compressed("results/dec_encoder_weights.npz", *encoder.get_weights())
np.savez("results/dec_clustering.npz", centers=clustering_layer.get_weights()[0], alpha=np.array(1.0))
print("Saved deployment artifacts: results/dec_encoder_weights.npz and results/dec_clustering.npz")


metrics = pd.DataFrame({
    "Method": ["DEC"],
    "Clusters": [n_clusters],
    "Silhouette_Score": [silhouette],
    "Davies_Bouldin_Index": [davies_bouldin],
    "Calinski_Harabasz_Score": [calinski_harabasz]
})

metrics.to_csv(
    "results/dec_metrics.csv",
    index=False
)


# ============================================================
# 13. VISUALIZATION
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    original_df["x_coordinate"],
    original_df["y_coordinate"],
    c=dec_labels,
    s=25,
    alpha=0.7
)

plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.title("DEC Semantic User Grouping")
plt.grid(alpha=0.2)

plt.savefig(
    "results/dec_clusters.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nDEC results saved successfully!")
