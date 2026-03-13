import joblib
import pandas as pd

# Load results
classical = joblib.load("evaluation/classical_results.pkl")
ensemble = joblib.load("evaluation/ensemble_results.pkl")
deep = joblib.load("evaluation/deep_results.pkl")
hybrid = joblib.load("evaluation/hybrid_results.pkl")

all_results = []

# Classical models
for model, (acc, f1) in classical.items():
    all_results.append({
        "Category": "Classic",
        "Model": model,
        "Accuracy": acc,
        "F1 Score": f1
    })

# Ensemble models
for model, (acc, f1) in ensemble.items():
    all_results.append({
        "Category": "Ensemble",
        "Model": model,
        "Accuracy": acc,
        "F1 Score": f1
    })

# Deep learning models
for model, (acc, f1) in deep.items():
    all_results.append({
        "Category": "Deep Learning",
        "Model": model,
        "Accuracy": acc,
        "F1 Score": f1
    })

# Hybrid models
for model, (acc, f1) in hybrid.items():
    all_results.append({
        "Category": "Hybrid",
        "Model": model,
        "Accuracy": acc,
        "F1 Score": f1
    })

# Convert to DataFrame
df = pd.DataFrame(all_results)

# Sort by accuracy
df = df.sort_values(by="Accuracy", ascending=False)

print("\nModel Comparison")
print(df)

# Save comparison
df.to_csv("evaluation/model_comparison.csv", index=False)