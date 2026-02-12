import joblib
import numpy as np
import pandas as pd
import re
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, LSTM, Bidirectional, Concatenate, Dropout
from tensorflow.keras.utils import to_categorical

# --- Helper Functions for Preprocessing ---

def parse_sleep(val):
    if pd.isna(val): return 7.0 # Default
    if isinstance(val, (int, float)): return float(val)
    val = str(val).lower()
    if '<' in val: return 1.5
    if '>' in val: return 9.0
    # Extract numbers
    nums = [float(x) for x in re.findall(r'\d+', val)]
    if nums:
        return sum(nums) / len(nums)
    return 7.0

def parse_devices(df):
    # Create flags
    df['Has_Laptop'] = df.apply(lambda x: 1 if isinstance(x, str) and 'Laptop' in x else 0, axis=1) # Incorrect logic if applied to df, need column
    return df

# Load data
df = pd.read_csv("data/labeled_dataset.csv")

# 1. Target Encoding
le = LabelEncoder()
y = le.fit_transform(df['Risk_Status'])
y_cat = to_categorical(y)

# 2. Sequence Data (Semesters 1-4)
semester_cols = [
    'Semester 1(aggregate)','Semester 2(aggregate)','Semester 3(aggregate)','Semester 4(aggregate)'
]
for col in semester_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Save medians for imputation
medians = df[semester_cols].median()
df[semester_cols] = df[semester_cols].fillna(medians)
X_seq = df[semester_cols].values.reshape(-1, 4, 1)

# 3. Static Data (Profile)
# Features to use: Gender, Study hours, Attendance, Backlogs, Sleep, etc.
# We need to clean them first.

# Create a copy for static features
static_df = df.copy()

# -- Clean Sleep --
static_df['Sleep_Numeric'] = static_df['Sleep hours'].apply(parse_sleep)

# -- Clean Categoricals --
# Gender: 1=Male, 0=Female (Arbitrary, just consistent)
# Assuming dataset has 'Male', 'Female' or 0, 1. Checks needed.
# Converting to numeric if not already
if static_df['Gender'].dtype == 'object':
    static_df['Gender_Num'] = static_df['Gender'].apply(lambda x: 1 if str(x).lower().startswith('m') or str(x)=='1' else 0)
else:
     static_df['Gender_Num'] = static_df['Gender']

# Yes/No cols
yes_no_cols = ['Internet access(yes/no)', 'Extracurricular activities(yes/no)', 'Library Usage']
for col in yes_no_cols:
    if col in static_df.columns:
        static_df[col] = static_df[col].apply(lambda x: 1 if str(x).lower() in ['yes', '1', 'true'] else 0)
    else:
        static_df[col] = 0

# Levels (Low/Medium/High)
level_map = {'low': 0, 'medium': 1, 'high': 2, '0': 0, '1': 1, '2': 2}
for col in ['Classroom engagement level', 'Peer interaction level']:
    static_df[col] = static_df[col].astype(str).str.lower().map(level_map).fillna(1) # Default Medium

# Devices
# The dataset has multiple columns or a combined column? 
# Looking at view_file output, it seems to have one-hot encoded columns like "Device used for studing_Laptops, Mobile".
# Let's clean the column names or checking the raw column if it exists, or combine the existing OHE columns.
# Actually, the dataset view showed columns like "Device used for studing_Laptops".
# Let's consolidate if possible, or just use `Study hours`, `Attendance`, `Backlogs` and the cleaned ones.

# Let's create specific flags based on column presence or content if original column exists.
# The `view_file` showed one-hot cols in the CSV header! 
# "Device used for studing_Laptops, Mobile", ...
# This implies the raw CSV might already be processed or has these specific string columns?
# Wait, line 1 of CSV: "Device used for studing_Laptops, Mobile", "Device used for studing_Laptops, Mobile, Text books"...
# These look like Categorical Levels that were OHE'd or are just distinct string values in a column?
# Actually, standard pandas read_csv would parse "Device used..." as a column name.
# Let's look for known device keywords in ALL columns that start with "Device used".

device_cols = [c for c in df.columns if 'Device used' in c]
# We will create our own simplified flags:
static_df['Use_Laptop'] = 0
static_df['Use_Mobile'] = 0
static_df['Use_Textbooks'] = 0

for idx, row in static_df.iterrows():
    # Check all device cols, if any is true/1, check its name for keywords
    # OR if there's a single column with comma separated values.
    # The CSV header suggests One-Hot style but with complex names.
    # Let's just check if 'Laptop' is in ANY of the device columns that are True/1 for that row.
    has_laptop = 0
    has_mobile = 0
    has_text = 0
    
    for col in device_cols:
        val = row[col]
        # If value is True/1 or it's the column name itself that matters (if it's OHE).
        # Assuming these are boolean/binary columns?
        # If the content allows 'True' or 1.
        if str(val) in ['True', '1', '1.0']:
            if 'Laptop' in col: has_laptop = 1
            if 'Mobile' in col: has_mobile = 1
            if 'Text' in col: has_text = 1
            
    static_df.at[idx, 'Use_Laptop'] = has_laptop
    static_df.at[idx, 'Use_Mobile'] = has_mobile
    static_df.at[idx, 'Use_Textbooks'] = has_text

# Select Final Static Features
static_features = [
    'Gender_Num', 'Study hours', 'Attendance percentage', 'No. of backlogs',
    'Sleep_Numeric', 'Internet access(yes/no)', 'Extracurricular activities(yes/no)',
    'Library Usage', 'Classroom engagement level', 'Peer interaction level',
    'Use_Laptop', 'Use_Mobile', 'Use_Textbooks'
]

# Ensure numeric
for col in static_features:
    static_df[col] = pd.to_numeric(static_df[col], errors='coerce').fillna(0)

X_static = static_df[static_features].values

# Scale Static Features
scaler_static = StandardScaler()
X_static = scaler_static.fit_transform(X_static)

# Validation Split
# Stratify based on y
indices = np.arange(len(y))
X_seq_train, X_seq_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X_seq, y, indices, test_size=0.2, random_state=42, stratify=y
)
X_static_train = X_static[idx_train]
X_static_test = X_static[idx_test]
y_cat_train = y_cat[idx_train]
y_cat_test = y_cat[idx_test]

# 4. Build Hybrid Model (Multi-Input)

# Sequence Input Branch (BiLSTM)
input_seq = Input(shape=(4, 1), name='seq_input')
x_seq = Bidirectional(LSTM(32, return_sequences=False))(input_seq)
x_seq = Dense(16, activation='relu')(x_seq)

# Static Input Branch (Dense)
input_static = Input(shape=(len(static_features),), name='static_input')
x_static = Dense(32, activation='relu')(input_static)
x_static = Dropout(0.2)(x_static)
x_static = Dense(16, activation='relu')(x_static)

# Combine
combined = Concatenate()([x_seq, x_static])
z = Dense(32, activation='relu')(combined)
z = Dropout(0.2)(z)
output = Dense(2, activation='softmax')(z)

model = Model(inputs=[input_seq, input_static], outputs=output)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Training Multi-Input Hybrid Model...")
history = model.fit(
    [X_seq_train, X_static_train], y_cat_train,
    validation_data=([X_seq_test, X_static_test], y_cat_test),
    epochs=30, batch_size=32, verbose=1
)

# Evaluate
loss, acc = model.evaluate([X_seq_test, X_static_test], y_cat_test, verbose=0)
print(f"Test Accuracy: {acc:.4f}")

# 5. Save Artifacts
print("Saving artifacts...")
model.save("models/bilstm_hybrid.keras")
joblib.dump(scaler_static, "models/scaler_static.pkl")
joblib.dump(le, "models/label_encoder_hybrid.pkl")
joblib.dump(medians, "models/imputer_medians.pkl")
joblib.dump(static_features, "models/static_feature_names.pkl") # Save feature names for checking

print("All artifacts saved successfully.")

