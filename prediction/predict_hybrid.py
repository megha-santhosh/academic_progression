import joblib
import numpy as np
import pandas as pd
import re
from tensorflow.keras.models import load_model

def parse_sleep(val):
    if pd.isna(val): return 7.0
    if isinstance(val, (int, float)): return float(val)
    val = str(val).lower()
    if '<' in val: return 1.5
    if '>' in val: return 9.0
    nums = [float(x) for x in re.findall(r'\d+', val)]
    if nums:
        return sum(nums) / len(nums)
    return 7.0

def predict_academic_progression(input_data):
    """
    Predicts risk using Multi-Input Hybrid Model (BiLSTM + Dense).
    """
    # 1. Load Artifacts
    try:
        model = load_model("models/bilstm_hybrid.keras")
        scaler_static = joblib.load("models/scaler_static.pkl")
        le = joblib.load("models/label_encoder_hybrid.pkl")
        medians = joblib.load("models/imputer_medians.pkl")
        static_features = joblib.load("models/static_feature_names.pkl")
    except Exception as e:
        return [f"Error loading models: {e}"], [0.0]

    # 2. Preprocess
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    else:
        df = input_data.copy()

    # --- Sequence Data ---
    semester_cols = [
        'Semester 1(aggregate)','Semester 2(aggregate)','Semester 3(aggregate)','Semester 4(aggregate)'
    ]
    for col in semester_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Fill missing sequence data
    df[semester_cols] = df[semester_cols].fillna(medians[semester_cols])
    X_seq = df[semester_cols].values.reshape(-1, 4, 1)

    # --- Static Data ---
    # Sleep
    if 'Sleep Hours' in df.columns:
        df['Sleep_Numeric'] = df['Sleep Hours'].apply(parse_sleep)
    else:
        df['Sleep_Numeric'] = 7.0

    # Gender
    if 'Gender' in df.columns:
        df['Gender_Num'] = df['Gender'].apply(lambda x: 1 if str(x).lower().startswith('m') else 0)
    else:
        df['Gender_Num'] = 0

    # Yes/No
    map_yes_no = lambda x: 1 if str(x).lower() in ['yes', '1', 'true'] else 0
    
    # Mapping App keys to Model keys
    df['Internet access(yes/no)'] = df.get('Internet Access', 'No').apply(map_yes_no)
    df['Extracurricular activities(yes/no)'] = df.get('Extracurriculars', 'No').apply(map_yes_no)
    df['Library Usage'] = df.get('Library Usage', 'No').apply(map_yes_no)

    # Levels
    level_map = {'low': 0, 'medium': 1, 'high': 2}
    df['Classroom engagement level'] = df.get('Classroom Engagement', 'Medium').astype(str).str.lower().map(level_map).fillna(1)
    df['Peer interaction level'] = df.get('Peer Interaction', 'Medium').astype(str).str.lower().map(level_map).fillna(1)

    # Devices
    # App sends list of strings e.g. ["Laptops", "Mobile"]
    # Model expects 'Use_Laptop', 'Use_Mobile', 'Use_Textbooks'
    df['Use_Laptop'] = 0
    df['Use_Mobile'] = 0
    df['Use_Textbooks'] = 0

    def parse_app_devices(val):
        laptop, mobile, text = 0, 0, 0
        if isinstance(val, list):
            s_val = " ".join(val).lower()
        else:
            s_val = str(val).lower()

        if 'laptop' in s_val: laptop = 1
        if 'mobile' in s_val: mobile = 1
        if 'text' in s_val: text = 1
        return laptop, mobile, text

    # Apply to each row
    # Handle single row case explicitly for clarity, though apply would work
    if len(df) == 1:
        l, m, t = parse_app_devices(df.iloc[0].get('Devices Used', []))
        df['Use_Laptop'] = l
        df['Use_Mobile'] = m
        df['Use_Textbooks'] = t
    else:
         # Vectorized for batch (not strictly needed for single prediction but good practice)
         processed_devs = df['Devices Used'].apply(parse_app_devices)
         df['Use_Laptop'] = [x[0] for x in processed_devs]
         df['Use_Mobile'] = [x[1] for x in processed_devs]
         df['Use_Textbooks'] = [x[2] for x in processed_devs]

    # Map other direct numeric columns
    # Model: 'Study hours', 'Attendance percentage', 'No. of backlogs'
    # App: 'Study Hours', 'Attendance', 'Backlogs'
    df['Study hours'] = pd.to_numeric(df.get('Study Hours', 0), errors='coerce')
    df['Attendance percentage'] = pd.to_numeric(df.get('Attendance', 0), errors='coerce')
    df['No. of backlogs'] = pd.to_numeric(df.get('Backlogs', 0), errors='coerce')

    # Ensure all static_features exist in df, fill with 0 if missing
    for col in static_features:
        if col not in df.columns:
            df[col] = 0.0 # Default to 0 for missing static features

    # Ensure all features exist and are ordered correctly
    X_static = df[static_features].values
    X_static = scaler_static.transform(X_static)

    # 3. Predict
    # Model expects [X_seq, X_static]
    pred_probs = model.predict([X_seq, X_static], verbose=0)
    pred_idx = np.argmax(pred_probs, axis=1)
    
    pred_label = le.inverse_transform(pred_idx)
    confidence = np.max(pred_probs, axis=1)
    
    return pred_label, confidence

if __name__ == "__main__":
    # Test Sample
    sample = {
        'Semester 1(aggregate)': 45.0,
        'Semester 2(aggregate)': 42.0,
        'Semester 3(aggregate)': 40.0,
        'Semester 4(aggregate)': 38.0,
        'Gender': 'Male',
        'Study Hours': 2.0,
        'Attendance': 60,
        'Backlogs': 2,
        'Sleep Hours': '4-6 hrs',
        'Internet Access': 'Yes',
        'Extracurriculars': 'No',
        'Library Usage': 'No',
        'Classroom Engagement': 'Low',
        'Peer Interaction': 'Low',
        'Devices Used': ['Mobile']
    }
    
    print(f"Testing Risk Student: {sample}")
    lbl, conf = predict_academic_progression(sample)
    print(f"Result: {lbl[0]} ({conf[0]:.4f})")
    
    sample_safe = {
        'Semester 1(aggregate)': 85.0,
        'Semester 2(aggregate)': 82.0,
        'Semester 3(aggregate)': 80.0,
        'Semester 4(aggregate)': 88.0,
        'Gender': 'Female',
        'Study Hours': 6.0,
        'Attendance': 90,
        'Backlogs': 0,
        'Sleep Hours': '6-8 hrs',
        'Internet Access': 'Yes',
        'Extracurriculars': 'Yes',
        'Library Usage': 'Yes',
        'Classroom Engagement': 'High',
        'Peer Interaction': 'High',
        'Devices Used': ['Laptops', 'Text books']
    }
    
    print(f"\nTesting Safe Student: {sample_safe}")
    lbl, conf = predict_academic_progression(sample_safe)
    print(f"Result: {lbl[0]} ({conf[0]:.4f})")
