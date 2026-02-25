import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model, Model


def predict_academic_progression(input_data):
    """
    Predicts academic risk using:
    BiLSTM (feature extraction) + XGBoost (classification)

    Handles partially completed semesters safely.
    """

    # ======================================================
    # LOAD MODELS
    # ======================================================
    try:
        bilstm_model = load_model("models/bilstm_feature_extractor.keras")
        xgb_model = joblib.load("models/xgb_bilstm.pkl")
        le = joblib.load("models/label_encoder.pkl")
    except Exception as e:
        return [f"Error loading models: {e}"], [0.0]

    # ======================================================
    # PREPARE INPUT
    # ======================================================
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    else:
        df = input_data.copy()

    semester_cols = [
        'Semester 1(aggregate)','Semester 2(aggregate)',
        'Semester 3(aggregate)','Semester 4(aggregate)',
        'Semester 5(aggregate)','Semester 6(aggregate)',
        'Semester 7(aggregate)','Semester 8(aggregate)'
    ]

    # Ensure all semester columns exist
    for col in semester_cols:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # ======================================================
    # SMART MISSING HANDLING
    # ======================================================
    # Replace 0 with NaN (since 0 usually means not completed)
    for col in semester_cols:
        df[col] = df[col].replace(0, np.nan)

    # Fill missing semesters with row mean (student's average)
    row_means = df[semester_cols].mean(axis=1)
    df[semester_cols] = df[semester_cols].apply(
        lambda row: row.fillna(row.mean()), axis=1
    )

    # If entire row is empty, fallback to 50%
    df[semester_cols] = df[semester_cols].fillna(50)

    # ======================================================
    # RESHAPE FOR BiLSTM
    # ======================================================
    X_seq = df[semester_cols].values.reshape(-1, 8, 1)

    # ======================================================
    # FEATURE EXTRACTION FROM BiLSTM
    # ======================================================
    feature_model = Model(
        inputs=bilstm_model.inputs,
        outputs=bilstm_model.layers[-2].output
    )

    X_features = feature_model.predict(X_seq, verbose=0)

    # ======================================================
    # XGBOOST PREDICTION
    # ======================================================
    pred_probs = xgb_model.predict_proba(X_features)
    pred_idx = np.argmax(pred_probs, axis=1)

    pred_label = le.inverse_transform(pred_idx)
    confidence = np.max(pred_probs, axis=1)

    return pred_label, confidence


# ==========================================================
# TESTING
# ==========================================================
if __name__ == "__main__":

    # Partially completed student
    sample_partial = {
        'Semester 1(aggregate)': 85.0,
        'Semester 2(aggregate)': 82.0,
        'Semester 3(aggregate)': 80.0,
        'Semester 4(aggregate)': 88.0,
        'Semester 5(aggregate)': 0.0,
        'Semester 6(aggregate)': 0.0,
        'Semester 7(aggregate)': 0.0,
        'Semester 8(aggregate)': 0.0,
    }

    lbl, conf = predict_academic_progression(sample_partial)
    print(f"Prediction (Partial): {lbl[0]} | Confidence: {conf[0]:.4f}")

    # Fully completed student
    sample_full = {
        'Semester 1(aggregate)': 75.0,
        'Semester 2(aggregate)': 72.0,
        'Semester 3(aggregate)': 70.0,
        'Semester 4(aggregate)': 68.0,
        'Semester 5(aggregate)': 65.0,
        'Semester 6(aggregate)': 63.0,
        'Semester 7(aggregate)': 60.0,
        'Semester 8(aggregate)': 58.0,
    }

    lbl, conf = predict_academic_progression(sample_full)
    print(f"Prediction (Full): {lbl[0]} | Confidence: {conf[0]:.4f}")