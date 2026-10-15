"""Predict a new user using saved training-time preprocessing and DEC artifacts."""
import argparse
import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Model, layers

RESULTS = "results"
NUMERIC = ["x_coordinate", "y_coordinate", "channel_condition",
           "emergency_priority", "delay_requirement", "semantic_similarity"]
FIELDS = ["x_coordinate", "y_coordinate", "channel_condition",
          "emergency_priority", "requested_data_type", "delay_requirement",
          "semantic_similarity"]


def student_t_assignments(z, centers, alpha=1.0):
    distances = np.sum((z[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    q = (1.0 + distances / alpha) ** (-(alpha + 1.0) / 2.0)
    return q / q.sum(axis=1, keepdims=True)


def predict(values):
    missing = set(FIELDS) - set(values)
    if missing:
        raise ValueError(f"Missing input fields: {', '.join(sorted(missing))}")
    record = dict(values)
    for field in NUMERIC:
        record[field] = float(record[field])
    record["requested_data_type"] = str(record["requested_data_type"])

    preprocessor = joblib.load(f"{RESULTS}/preprocessor.joblib")
    encoder_input = layers.Input(shape=(10,))
    hidden = layers.Dense(32, activation="relu")(encoder_input)
    hidden = layers.Dense(16, activation="relu")(hidden)
    latent = layers.Dense(5, activation="linear", name="latent")(hidden)
    encoder = Model(encoder_input, latent)
    weights = np.load(f"{RESULTS}/dec_encoder_weights.npz")
    encoder.set_weights([weights[key] for key in sorted(weights.files, key=lambda key: int(key.split("_")[-1]))])
    model_data = np.load(f"{RESULTS}/dec_clustering.npz")
    centers, alpha = model_data["centers"], float(model_data["alpha"])
    transformed = preprocessor.transform(pd.DataFrame([record], columns=FIELDS)).astype("float32")
    latent = encoder.predict(transformed, verbose=0)
    probabilities = student_t_assignments(latent, centers, alpha)[0]
    cluster = int(np.argmax(probabilities))

    characteristics = pd.read_csv(f"{RESULTS}/dec_cluster_analysis.csv").set_index("dec_cluster")
    row = characteristics.loc[cluster]
    group = str(row.get("Group_Name", "General"))
    mode_type = str(row.get("Most_Common_Data_Type", group))
    explanation = (f"This cluster's most common requested data type is {mode_type}; "
                   f"its mean emergency priority is {row['Avg_Priority']:.2f}, "
                   f"mean delay requirement is {row['Avg_Delay']:.2f}, and "
                   f"mean semantic similarity is {row['Avg_Semantic_Similarity']:.2f}. "
                   "These are observed cluster characteristics, not a resource-allocation decision.")
    output = {**record, "predicted_cluster": cluster, "interpreted_group": group,
              "cluster_confidence": float(probabilities[cluster]), "explanation": explanation}
    os.makedirs(RESULTS, exist_ok=True)
    pd.DataFrame([output]).to_csv(f"{RESULTS}/new_user_prediction.csv", index=False)
    return output


def main():
    parser = argparse.ArgumentParser(description="Predict a user's DEC semantic group using saved artifacts.")
    for field in FIELDS:
        parser.add_argument(f"--{field}", required=True, help=f"User {field}")
    args = vars(parser.parse_args())
    result = predict(args)
    print("New-user DEC prediction")
    print("Input values:")
    for field in FIELDS:
        print(f"  {field}: {result[field]}")
    print(f"Predicted cluster: {result['predicted_cluster']}")
    print(f"Interpreted group: {result['interpreted_group']}")
    print(f"DEC assignment confidence: {result['cluster_confidence']:.4f}")
    print(f"Explanation: {result['explanation']}")
    print(f"Saved: {RESULTS}/new_user_prediction.csv")


if __name__ == "__main__":
    main()
