# social_ml/src/train_model.py
import os, joblib, pandas as pd, sys
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from preprocess import clean_text

ROOT = os.path.dirname(__file__)
DATA_PATH = os.path.join(ROOT, '..', 'data', 'tweets_demo.csv')
MODEL_OUT = os.path.join(ROOT, '..', 'models', 'disaster_model.joblib')

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df = df.dropna(subset=['text','label'])
    df['text_clean'] = df['text'].apply(clean_text)
    return df

def train():
    df = load_data()
    X = df['text_clean']
    y = df['label'].astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=8000, ngram_range=(1,2))),
        ('clf', LogisticRegression(solver='liblinear', class_weight='balanced', max_iter=1000))
    ])
    pipe.fit(Xtr, ytr)
    preds = pipe.predict(Xte)
    print("Classification report:")
    print(classification_report(yte, preds))
    print("Confusion matrix:")
    print(confusion_matrix(yte, preds))
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    joblib.dump(pipe, MODEL_OUT)
    print("Saved model to", MODEL_OUT)

if __name__ == "__main__":
    train()
