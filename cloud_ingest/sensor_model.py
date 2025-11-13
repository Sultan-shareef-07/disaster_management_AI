# cloud_ingest/sensor_model.py
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

# normalize model output path so it's absolute regardless of CWD
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
MODEL_OUT = os.path.join(MODEL_DIR, 'sensor_iforest.joblib')

def featurize_window(df):
    """Return summary features for expected sensor columns; robust to non-numeric values."""
    feats = {}
    for c in ['vibration', 'flame', 'water']:
        if c in df.columns:
            # coerce to numeric, drop non-numeric rows
            vals = pd.to_numeric(df[c], errors='coerce').dropna().astype(float).values
            if len(vals) > 0:
                feats[c + '_mean'] = float(np.mean(vals))
                feats[c + '_std'] = float(np.std(vals, ddof=0))
                feats[c + '_max'] = float(np.max(vals))
            else:
                feats[c + '_mean'] = 0.0
                feats[c + '_std'] = 0.0
                feats[c + '_max'] = 0.0
        else:
            feats[c + '_mean'] = 0.0
            feats[c + '_std'] = 0.0
            feats[c + '_max'] = 0.0
    return feats

def train_iforest(csv_path, out=MODEL_OUT, window_size=10):
    """
    Train an IsolationForest on windowed aggregate features.
    csv_path may be an absolute path or relative to repository root (../demo_data/...).
    """
    # resolve path if string
    if isinstance(csv_path, str) and not os.path.isabs(csv_path):
        csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', csv_path))
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Failed to read CSV '{csv_path}': {e}")
        return

    windows = []
    for i in range(0, len(df), window_size):
        w = df.iloc[i:i + window_size]
        feats = featurize_window(w)
        windows.append(feats)

    X = pd.DataFrame(windows).fillna(0.0)
    if X.empty:
        print("No windows generated from input data; aborting training.")
        return

    model = IsolationForest(contamination=0.02, random_state=42)
    model.fit(X)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    joblib.dump((model, X.columns.tolist()), out)
    print("Saved sensor model:", out)

def predict_window(window_df, model_tuple=None):
    """
    Accepts a DataFrame (window_df) and returns (is_anomaly: bool, anomaly_score: float).
    If model_tuple is None, loads the model from MODEL_OUT.
    The returned score is positive when more anomalous (higher = more anomalous).
    """
    if model_tuple is None:
        if not os.path.exists(MODEL_OUT):
            raise FileNotFoundError(f"Sensor model not found at '{MODEL_OUT}'. Train model first.")
        model, cols = joblib.load(MODEL_OUT)
    else:
        model, cols = model_tuple

    feats = featurize_window(window_df)
    X = pd.DataFrame([feats]).reindex(columns=cols, fill_value=0.0)

    # decision_function: higher -> more normal, lower -> more anomalous
    raw_score = model.decision_function(X)[0]
    is_anom = model.predict(X)[0] == -1
    # invert raw_score to produce positive anomaly-score (larger -> more anomalous)
    anomaly_score = float(-raw_score)
    return bool(is_anom), anomaly_score

if __name__ == '__main__':
    demo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'demo_data', 'sensor_demo.csv'))
    train_iforest(demo_path)
