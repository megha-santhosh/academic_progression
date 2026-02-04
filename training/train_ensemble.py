import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# Load data
X_train = joblib.load("data/X_train.pkl")
X_test = joblib.load("data/X_test.pkl")
y_train = joblib.load("data/y_train.pkl")
y_test = joblib.load("data/y_test.pkl")

# Drop columns that are all NaN
mask = ~np.isnan(X_train).all(axis=0)
X_train = X_train[:, mask]
X_test = X_test[:, mask]

# Impute NaN values
imputer = SimpleImputer(strategy="median")
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Encode labels for XGBoost/LightGBM
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

models = {
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=200, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
    "XGBoost": XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42),
    "LightGBM": LGBMClassifier(random_state=42)
}

results = {}

for name, model in models.items():
    if name in ["XGBoost", "LightGBM"]:
        model.fit(X_train, y_train_enc)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test_enc, y_pred)
        f1 = f1_score(y_test_enc, y_pred, average="macro")
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")

    results[name] = (acc, f1)
    joblib.dump(model, f"models/{name}.pkl")
    print(f"{name}: Accuracy={acc:.4f}, F1={f1:.4f}")

joblib.dump(results, "evaluation/ensemble_results.pkl")
