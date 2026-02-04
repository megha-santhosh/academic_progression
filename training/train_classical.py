import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
import numpy as np


X_train = joblib.load("data/X_train.pkl")
X_test = joblib.load("data/X_test.pkl")
y_train = joblib.load("data/y_train.pkl")
y_test = joblib.load("data/y_test.pkl")

# Drop columns that are all NaN
mask = ~np.isnan(X_train).all(axis=0)
X_train = X_train[:, mask]
X_test = X_test[:, mask]

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "SVM": SVC(kernel='rbf', probability=True)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, pos_label="At Safe")

    results[name] = (acc, f1)

    joblib.dump(model, f"models/{name}.pkl")

    print(f"{name}: Accuracy={acc:.4f}, F1={f1:.4f}")

joblib.dump(results, "evaluation/classical_results.pkl")
