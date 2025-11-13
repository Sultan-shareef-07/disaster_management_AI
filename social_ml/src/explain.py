# social_ml/src/explain.py
import joblib, numpy as np, os
MODEL = os.path.join(os.path.dirname(__file__), '..', 'models', 'disaster_model.joblib')

def top_features(n=10):
    pipe = joblib.load(MODEL)
    vec = pipe.named_steps['tfidf']
    clf = pipe.named_steps['clf']
    feats = vec.get_feature_names_out()
    coefs = clf.coef_[0]
    pos_idx = np.argsort(coefs)[-n:][::-1]
    neg_idx = np.argsort(coefs)[:n]
    return {'positive': [(feats[i], float(coefs[i])) for i in pos_idx],
            'negative': [(feats[i], float(coefs[i])) for i in neg_idx]}

if __name__=='__main__':
    print(top_features(15))
