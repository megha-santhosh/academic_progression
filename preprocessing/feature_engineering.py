import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
df = pd.read_csv("data/labeled_dataset.csv")

y = df['Risk_Status']
X = df.drop([
    'Risk_Status', 'Trend', 'slope',
    'Semester 1(aggregate)','Semester 2(aggregate)',
    'Semester 3(aggregate)','Semester 4(aggregate)',
    'Semester 5(aggregate)','Semester 6(aggregate)',
    'Semester 7(aggregate)','Semester 8(aggregate)'
], axis=1)


# Convert any remaining categorical columns to numeric if necessary
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = (
            X[col].astype(str)
            .str.extract(r'(\d+\.?\d*)')[0]
            .astype(float)
        )


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save everything
joblib.dump(X_train_scaled, "data/X_train.pkl")
joblib.dump(X_test_scaled, "data/X_test.pkl")
joblib.dump(y_train, "data/y_train.pkl")
joblib.dump(y_test, "data/y_test.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("Feature engineering completed and files saved.")
