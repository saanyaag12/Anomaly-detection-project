import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# Generate synthetic dataset
np.random.seed(42)
normal_data = np.random.randn(100, 2)
anomalies = np.random.uniform(low=-6, high=6, size=(10, 2))

# Combine into one dataset
data = np.concatenate([normal_data, anomalies], axis=0)
df = pd.DataFrame(data, columns=["feature1", "feature2"])

# Fit Isolation Forest
model = IsolationForest(contamination=0.1, random_state=42)
df["anomaly"] = model.fit_predict(df[["feature1", "feature2"]])

# Map anomaly labels: -1 → anomaly, 1 → normal
df["anomaly"] = df["anomaly"].map({1: 0, -1: 1})

# Plot the results
plt.figure(figsize=(8, 6))
plt.scatter(df["feature1"], df["feature2"], c=df["anomaly"], cmap="coolwarm", edgecolor="k")
plt.title("Anomaly Detection using Isolation Forest")
plt.xlabel("feature1")
plt.ylabel("feature2")
plt.show()
