# Student Risk Prediction System

A hybrid machine learning system to classify students as **At Risk** or **At Safe** using academic trends and behavioral features.

## Features
- Rule-based label generation
- Classical, Ensemble, Deep & Hybrid models
- Data leakage handling
- Model comparison
- Streamlit ready

## Models Used
- Logistic Regression, KNN, SVM
- Random Forest, AdaBoost, XGBoost, LightGBM
- CNN, LSTM, BiLSTM
- CNN+XGBoost, BiLSTM+XGBoost, Autoencoder+RF

## How to Run
```bash
pip install -r requirements.txt
python preprocessing/clean_data.py
python preprocessing/label_generation.py
python preprocessing/feature_engineering.py
python training/train_classical.py
python training/train_ensemble.py
python training/train_deep.py
python training/train_hybrid.py
python evaluation/compare_models.py
