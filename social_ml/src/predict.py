# social_ml/src/predict.py
import os, joblib
from preprocess import clean_text

MODEL = os.path.join(os.path.dirname(__file__), '..', 'models', 'disaster_model.joblib')

def load_model():
    return joblib.load(MODEL)

def predict_text(text, model=None):
    if model is None:
        model = load_model()
    text_c = clean_text(text)
    label = int(model.predict([text_c])[0])
    prob = float(model.predict_proba([text_c])[0].max())
    return label, prob

if __name__=='__main__':
    m = load_model()
    for t in ["Huge flood in my area, need help", "Beautiful day at the beach"]:
        print(t, predict_text(t, m))
