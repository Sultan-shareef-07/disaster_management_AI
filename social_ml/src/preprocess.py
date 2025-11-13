# social_ml/src/preprocess.py
import re, emoji
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords', quiet=True)
STOPWORDS = set(stopwords.words('english'))

def remove_emoji(text):
    return emoji.replace_emoji(text, replace='')

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = remove_emoji(text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#','', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [t for t in text.split() if t not in STOPWORDS and len(t) > 1]
    return ' '.join(tokens)
