import pandas as pd

df = pd.read_csv("data/disaster_users.csv")
print("Shape:", df.shape)
print("\nMissing values:")
print(df.isna().sum())
print("\nFirst rows:")
print(df.head().to_string(index=False))
