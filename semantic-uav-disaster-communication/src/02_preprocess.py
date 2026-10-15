import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load dataset
df = pd.read_csv("data/disaster_users.csv")

# Separate user ID
user_ids = df["user_id"]

# Features used for clustering
X = df.drop(columns=["user_id"])

# Numerical features
numerical_features = [
    "x_coordinate",
    "y_coordinate",
    "channel_condition",
    "emergency_priority",
    "delay_requirement",
    "semantic_similarity"
]

# Categorical feature
categorical_features = [
    "requested_data_type"
]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
         categorical_features)
    ]
)

# Transform the data
X_processed = preprocessor.fit_transform(X)

print("Original dataset shape:", X.shape)
print("Processed feature shape:", X_processed.shape)
print("Preprocessing completed successfully!")

# Save processed data
processed_df = pd.DataFrame(X_processed)

processed_df.to_csv(
    "data/processed_users.csv",
    index=False
)

os.makedirs("results", exist_ok=True)
joblib.dump(preprocessor, "results/preprocessor.joblib")
print("Preprocessing artifact saved to results/preprocessor.joblib")

print("Processed dataset saved to data/processed_users.csv")
