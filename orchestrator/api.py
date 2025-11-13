# orchestrator/api.py
from flask import Flask, request, jsonify
import os, joblib, time, json, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from social_ml.src.predict import predict_text, load_model as load_text_model
from cloud_ingest.sensor_model import predict_window

app = Flask(__name__)
TEXT_MODEL = None

def load_models():
    global TEXT_MODEL
    TEXT_MODEL = load_text_model()

@app.route('/predict/text', methods=['POST'])
def predict_text_route():
    data = request.json or {}
    text = data.get('text','')
    if not text:
        return jsonify({'error':'text required'}), 400
    label, prob = predict_text(text, TEXT_MODEL)
    return jsonify({'label': label, 'confidence': prob})

@app.route('/predict/sensor', methods=['POST'])
def predict_sensor_route():
    payload = request.json or {}
    # expecting list of samples or single sample -> unify to small DataFrame
    import pandas as pd
    df = pd.DataFrame(payload.get('window', []))
    if df.empty:
        return jsonify({'error':'window list required (list of samples)'}), 400
    anom, score = predict_window(df)
    return jsonify({'alert': anom, 'score': score})

if __name__=='__main__':
    load_models()
    app.run(host='0.0.0.0', port=5000, debug=True)
