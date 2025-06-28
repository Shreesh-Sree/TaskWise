import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

# Synthetic Data Generation
np.random.seed(42)
data = []
for _ in range(1000):
    imp = np.random.randint(1, 6)
    eff = np.random.randint(1, 11)
    days = np.random.randint(0, 10)
    label = "High" if imp >= 4 and days <= 2 else "Medium" if imp >= 3 and days <= 5 else "Low"
    data.append([imp, eff, days, label])

df = pd.DataFrame(data, columns=["Importance", "Effort", "Days_Left", "Priority"])
df["Priority"] = df["Priority"].map({"Low": 0, "Medium": 1, "High": 2})

X = df[["Importance", "Effort", "Days_Left"]]
y = df["Priority"]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y)

# Train Model
model = GradientBoostingClassifier()
model.fit(X_train, y_train)

# Save
joblib.dump(model, "model/super_model.pkl")
joblib.dump(scaler, "model/super_scaler.pkl")
