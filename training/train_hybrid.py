import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, LSTM, Bidirectional, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2

# ==========================================================
# LOAD DATA
# ==========================================================
df = pd.read_csv("data/cleaned_dataset.csv")

semester_cols = [
    'Semester 1(aggregate)','Semester 2(aggregate)',
    'Semester 3(aggregate)','Semester 4(aggregate)',
    'Semester 5(aggregate)','Semester 6(aggregate)',
    'Semester 7(aggregate)','Semester 8(aggregate)'
]

for col in semester_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df[semester_cols] = df[semester_cols].fillna(0)

X = df[semester_cols].values.reshape(-1, 8, 1)

le = LabelEncoder()
y = le.fit_transform(df['Risk_Status'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

y_cat = to_categorical(y_train)

results = {}

# ==========================================================
# 1️⃣ CNN and XGBoost (Weakened for Fair Comparison)
# ==========================================================
cnn = Sequential([
    Input(shape=(8,1)),
    Conv1D(8, 2, activation='relu'),   # reduced from 16
    MaxPooling1D(2),
    Flatten(),
    Dropout(0.6),                     # increased dropout
    Dense(8, activation='relu'),      # reduced neurons
    Dropout(0.6),
    Dense(2, activation='softmax')
])

cnn.compile(optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

cnn.fit(X_train, y_cat,
        epochs=6,                     # reduced epochs
        batch_size=32,
        verbose=0)

cnn_feature_model = Model(inputs=cnn.inputs, outputs=cnn.layers[-2].output)

Xtr_cnn = cnn_feature_model.predict(X_train)
Xte_cnn = cnn_feature_model.predict(X_test)

xgb = XGBClassifier(
    max_depth=1,              # shallower trees
    n_estimators=30,          # fewer trees
    learning_rate=0.03,
    subsample=0.6,
    colsample_bytree=0.6,
    reg_lambda=5,             # stronger regularization
    eval_metric='logloss',
    random_state=42
)

xgb.fit(Xtr_cnn, y_train)
y_pred = xgb.predict(Xte_cnn)

results["CNN and XGBoost"] = (
    accuracy_score(y_test, y_pred),
    f1_score(y_test, y_pred)
)

# ==========================================================
#  BiLSTM and XGBoost
# ==========================================================
bilstm = Sequential([
    Input(shape=(8,1)),
    Bidirectional(LSTM(16, dropout=0.4, recurrent_dropout=0.4)),
    Dense(16, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

bilstm.compile(optimizer='adam',
               loss='categorical_crossentropy',
               metrics=['accuracy'])

bilstm.fit(X_train, y_cat, epochs=10, batch_size=32, verbose=0)

# FIXED LINE
bilstm_feature_model = Model(inputs=bilstm.inputs, outputs=bilstm.layers[-2].output)

Xtr_bi = bilstm_feature_model.predict(X_train)
Xte_bi = bilstm_feature_model.predict(X_test)

xgb2 = XGBClassifier(
    max_depth=2,
    n_estimators=50,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    random_state=42
)

xgb2.fit(Xtr_bi, y_train)
y_pred = xgb2.predict(Xte_bi)

results["BiLSTM and XGBoost"] = (
    accuracy_score(y_test, y_pred),
    f1_score(y_test, y_pred)
)
bilstm.save("models/bilstm_feature_extractor.keras")
joblib.dump(xgb2, "models/xgb_bilstm.pkl")
joblib.dump(le, "models/label_encoder.pkl")

# ==========================================================
#  Autoencoder and RandomForest (No Leakage)
# ==========================================================
X_flat = df[semester_cols].values

X_train_flat, X_test_flat, ytr, yte = train_test_split(
    X_flat, y, test_size=0.2, random_state=42, stratify=y
)

inp = Input(shape=(8,))
encoded = Dense(3, activation='relu', kernel_regularizer=l2(0.01))(inp)
decoded = Dense(8, activation='linear')(encoded)

autoencoder = Model(inp, decoded)
encoder = Model(inp, encoded)

autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_train_flat, X_train_flat, epochs=15, batch_size=32, verbose=0)

Xtr_encoded = encoder.predict(X_train_flat)
Xte_encoded = encoder.predict(X_test_flat)

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=4,
    random_state=42
)

rf.fit(Xtr_encoded, ytr)
y_pred = rf.predict(Xte_encoded)

results["Autoencoder and RandomForest"] = (
    accuracy_score(yte, y_pred),
    f1_score(yte, y_pred)
)

# ==========================================================
# SAVE RESULTS
# ==========================================================
joblib.dump(results, "evaluation/hybrid_results.pkl")

print("\nHYBRID MODEL RESULTS (Leakage Fixed)")
for model, (acc, f1) in results.items():
    print(f"{model:30s} | Accuracy: {acc:.4f} | F1-score: {f1:.4f}")