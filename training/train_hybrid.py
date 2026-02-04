import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, LSTM, Bidirectional, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.utils import to_categorical

# Load labeled data
df = pd.read_csv("data/labeled_dataset.csv")

semester_cols = [
    'Semester 1(aggregate)','Semester 2(aggregate)','Semester 3(aggregate)',
    'Semester 4(aggregate)','Semester 5(aggregate)','Semester 6(aggregate)',
    'Semester 7(aggregate)','Semester 8(aggregate)'
]

# Numeric
for col in semester_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df[semester_cols] = df[semester_cols].fillna(df[semester_cols].median())

# SAFE: only early semesters as input
early_cols = semester_cols[:4]
X_seq = df[early_cols].values.reshape(-1, 4, 1)

# Labels
le = LabelEncoder()
y = le.fit_transform(df['Risk_Status'])

X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y, test_size=0.2, random_state=42, stratify=y
)

results = {}

# ---------- CNN + XGBoost ----------
y_cat = to_categorical(y_train)

cnn = Sequential([
    Input(shape=(4,1)),
    Conv1D(32, 2, activation='relu'),
    MaxPooling1D(2),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')
])

cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
cnn.fit(X_train, y_cat, epochs=10, batch_size=32, verbose=0)

cnn_feat_model = Model(inputs=cnn.inputs, outputs=cnn.layers[-2].output)
Xtr_cnn = cnn_feat_model.predict(X_train)
Xte_cnn = cnn_feat_model.predict(X_test)

xgb = XGBClassifier(eval_metric='logloss', random_state=42)
xgb.fit(Xtr_cnn, y_train)
y_pred = xgb.predict(Xte_cnn)

results["CNN+XGBoost"] = (
    accuracy_score(y_test, y_pred),
    f1_score(y_test, y_pred)
)

# ---------- BiLSTM + XGBoost ----------
bilstm = Sequential([
    Input(shape=(4,1)),
    Bidirectional(LSTM(32)),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')
])

bilstm.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
bilstm.fit(X_train, y_cat, epochs=10, batch_size=32, verbose=0)

bilstm_feat_model = Model(inputs=bilstm.inputs, outputs=bilstm.layers[-2].output)
Xtr_bi = bilstm_feat_model.predict(X_train)
Xte_bi = bilstm_feat_model.predict(X_test)

xgb2 = XGBClassifier(eval_metric='logloss', random_state=42)
xgb2.fit(Xtr_bi, y_train)
y_pred = xgb2.predict(Xte_bi)

results["BiLSTM+XGBoost"] = (
    accuracy_score(y_test, y_pred),
    f1_score(y_test, y_pred)
)

# ---------- Autoencoder + RandomForest ----------
X_flat = df[early_cols].values

inp = Input(shape=(4,))
encoded = Dense(2, activation='relu')(inp)
decoded = Dense(4, activation='linear')(encoded)
autoencoder = Model(inp, decoded)
encoder = Model(inp, encoded)

autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_flat, X_flat, epochs=20, batch_size=32, verbose=0)

X_encoded = encoder.predict(X_flat)

Xtr, Xte, ytr, yte = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(Xtr, ytr)
y_pred = rf.predict(Xte)

results["Autoencoder+RF"] = (
    accuracy_score(yte, y_pred),
    f1_score(yte, y_pred)
)

joblib.dump(results, "evaluation/hybrid_results.pkl")

print("HYBRID MODEL RESULTS")
for model, (acc, f1) in results.items():
    print(f"{model:20s} | Accuracy: {acc:.4f} | F1-score: {f1:.4f}")

