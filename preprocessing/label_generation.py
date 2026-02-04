import pandas as pd
import numpy as np

# Load cleaned data
df = pd.read_csv("data/cleaned_dataset.csv")

semester_cols = [
    'Semester 1(aggregate)','Semester 2(aggregate)','Semester 3(aggregate)',
    'Semester 4(aggregate)','Semester 5(aggregate)','Semester 6(aggregate)',
    'Semester 7(aggregate)','Semester 8(aggregate)'
]

# Force semester columns to numeric again (CSV loses dtypes)
for col in semester_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Fill any remaining NaNs with column median
df[semester_cols] = df[semester_cols].fillna(df[semester_cols].median())

# 1. Compute slope safely
def get_slope(row):
    y = row[semester_cols].astype(float).values

    # Replace any remaining NaN with row mean
    if np.isnan(y).any():
        y = np.nan_to_num(y, nan=np.nanmean(y))

    x = np.arange(1, 9)
    m, _ = np.polyfit(x, y, 1)
    return m

df['slope'] = df.apply(get_slope, axis=1)

# 2. Classify trend
def classify_trend(m):
    if m > 0.8:
        return 'Improving'
    elif m < -0.8:
        return 'Deteriorating'
    else:
        return 'Stable'

df['Trend'] = df['slope'].apply(classify_trend)

# 3. Average score
df['avg_score'] = df[semester_cols].mean(axis=1)

# 4. Risk status (your simplified rule)
def risk_label(row):
    if row['Trend'] == 'Deteriorating':
        return 'At Risk'
    else:
        return 'At Safe'

df['Risk_Status'] = df.apply(risk_label, axis=1)

# Save labeled dataset
df.to_csv("data/labeled_dataset.csv", index=False)

print("Labels created and saved as data/labeled_dataset.csv")
