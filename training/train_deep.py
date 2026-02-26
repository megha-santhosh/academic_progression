import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Conv1D, MaxPooling1D, Flatten, Input
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load labeled dataset
df = pd.read_csv("data/cleaned_dataset.csv")

semester_cols = [
    'Semester 1(aggregate)','Semester 2(aggregate)','Semester 3(aggregate)',
    'Semester 4(aggregate)','Semester 5(aggregate)','Semester 6(aggregate)',
    'Semester 7(aggregate)','Semester 8(aggregate)'
]

# Convert to numeric
for col in semester_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df[semester_cols] = df[semester_cols].fillna(df[semester_cols].median())

# Sequence data
X = df[semester_cols].values
X = X.reshape((X.shape[0], X.shape[1], 1))

# Encode labels
le = LabelEncoder()
y = le.fit_transform(df['Risk_Status'])
y_cat = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=42, stratify=y
)

# ----- CNN -----
cnn = Sequential([
    Conv1D(32, 2, activation='relu', input_shape=(8,1)),
    MaxPooling1D(2),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')
])

cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
cnn.fit(X_train, y_train, epochs=15, batch_size=32, verbose=0)
cnn_acc = cnn.evaluate(X_test, y_test, verbose=0)[1]

# ----- LSTM -----
lstm = Sequential([
    LSTM(32, input_shape=(8,1)),
    Dense(2, activation='softmax')
])

lstm.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
lstm.fit(X_train, y_train, epochs=15, batch_size=32, verbose=0)
lstm_acc = lstm.evaluate(X_test, y_test, verbose=0)[1]

# ----- BiLSTM -----
bilstm = Sequential([
    Bidirectional(LSTM(32), input_shape=(8,1)),
    Dense(2, activation='softmax')
])

bilstm.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
bilstm.fit(X_train, y_train, epochs=15, batch_size=32, verbose=0)
bilstm_acc = bilstm.evaluate(X_test, y_test, verbose=0)[1]

# ==========================================================
# ----- Autoencoder -----
# ==========================================================

# Use flat input (not sequence)
X_flat = df[semester_cols].values

X_train_flat, X_test_flat, y_train_flat, y_test_flat = train_test_split(
    X_flat, y_cat, test_size=0.2, random_state=42, stratify=y
)

# Build autoencoder
input_layer = Input(shape=(8,))
encoded = Dense(4, activation='relu')(input_layer)
decoded = Dense(8, activation='linear')(encoded)

autoencoder = Model(input_layer, decoded)
encoder = Model(input_layer, encoded)

autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_train_flat, X_train_flat, epochs=15, batch_size=32, verbose=0)

# Extract compressed features
X_train_encoded = encoder.predict(X_train_flat)
X_test_encoded = encoder.predict(X_test_flat)

# Classification layer on encoded features
auto_classifier = Sequential([
    Dense(16, activation='relu', input_shape=(4,)),
    Dense(2, activation='softmax')
])

auto_classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
auto_classifier.fit(X_train_encoded, y_train_flat, epochs=15, batch_size=32, verbose=0)

auto_acc = auto_classifier.evaluate(X_test_encoded, y_test_flat, verbose=0)[1]

# ==========================================================
# Save models
# ==========================================================
cnn.save("models/cnn_model.h5")
lstm.save("models/lstm_model.h5")
bilstm.save("models/bilstm_model.h5")
autoencoder.save("models/autoencoder_model.h5")

results = {
    "CNN": cnn_acc,
    "LSTM": lstm_acc,
    "BiLSTM": bilstm_acc,
    "Autoencoder": auto_acc
}

joblib.dump(results, "evaluation/deep_results.pkl")

print("\nDEEP MODEL RESULTS")
print(results)