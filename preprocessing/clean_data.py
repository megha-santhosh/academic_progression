import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load raw data
df = pd.read_csv("data/raw_dataset.csv")

print("Original shape:", df.shape)

# 1. Drop ID column
if 'Register Number' in df.columns:
    df.drop(columns=['Register Number'], inplace=True)

# 2. Handle missing values
for col in df.select_dtypes(include='number'):
    df[col].fillna(df[col].median(), inplace=True)

for col in df.select_dtypes(include='object'):
    df[col].fillna(df[col].mode()[0], inplace=True)

# 3. Encode binary categorical columns
le = LabelEncoder()

binary_cols = [
    'Gender',
    'Internet access(yes/no)',
    'Extracurricular activities(yes/no)'
]

for col in binary_cols:
    if col in df.columns:
        df[col] = le.fit_transform(df[col])

# 4. Ordinal encoding
ordinal_map = {'Low': 0, 'Medium': 1, 'High': 2}

ordinal_cols = [
    'Library Usage',
    'Classroom engagement level',
    'Peer interaction level'
]

for col in ordinal_cols:
    if col in df.columns:
        df[col] = df[col].map(ordinal_map)

# 5. One-hot encode device column
if 'Device used for studing' in df.columns:
    df = pd.get_dummies(df, columns=['Device used for studing'], drop_first=True)

# Save cleaned data
df.to_csv("data/cleaned_dataset.csv", index=False)

print("Cleaned data saved as data/cleaned_dataset.csv")
