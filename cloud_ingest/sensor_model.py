# cloud_ingest/sensor_model.py
import os, joblib, pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

MODEL_OUT = os.path.join(os.path.dirname(__file__), '..', 'models', 'sensor_iforest.joblib')

def featurize_window(df):
    feats = {}
    for c in ['vibration','flame','water']:
        if c in df.columns:
            vals = df[c].astype(float).values
            feats[c+'_mean'] = vals.mean() if len(vals)>0 else 0
            feats[c+'_std'] = vals.std() if len(vals)>0 else 0
            feats[c+'_max'] = vals.max() if len(vals)>0 else 0
    return feats

def train_iforest(csv_path, out=MODEL_OUT, window_size=10):
    df = pd.read_csv(csv_path)
    windows=[]
    for i in range(0, len(df), window_size):
        w = df.iloc[i:i+window_size]
        windows.append(featurize_window(w))
    X = pd.DataFrame(windows).fillna(0)
    model = IsolationForest(contamination=0.02, random_state=42)
    model.fit(X)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    joblib.dump((model, X.columns.tolist()), out)
    print("Saved sensor model:", out)

def predict_window(window_df, model_tuple=None):
    if model_tuple is None:
        model, cols = joblib.load(MODEL_OUT)
    else:
        model, cols = model_tuple
    feats = featurize_window(window_df)
    X = pd.DataFrame([feats]).reindex(columns=cols, fill_value=0)
    score = model.decision_function(X)[0]
    is_anom = model.predict(X)[0] == -1
    return bool(is_anom), float(-score)
