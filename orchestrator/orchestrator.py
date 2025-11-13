# orchestrator/orchestrator.py
import requests, time
TEXT_API = "http://localhost:5000/predict/text"
SENSOR_API = "http://localhost:5000/predict/sensor"

def fuse_and_decide(sensor_window, tweets):
    r = requests.post(SENSOR_API, json={'window': sensor_window}).json()
    sensor_alert = r.get('alert', False)
    sensor_score = r.get('score', 0.0)
    tweet_alerts = 0
    confidences = []
    for t in tweets:
        r2 = requests.post(TEXT_API, json={'text': t}).json()
        if r2.get('label',0)==1:
            tweet_alerts += 1
        confidences.append(r2.get('confidence',0))
    avg_conf = sum(confidences)/len(confidences) if confidences else 0
    final = False
    reasons=[]
    if sensor_alert:
        final = True; reasons.append('sensor_anomaly')
    if not final and sensor_score >= 0.6 and tweet_alerts >= 2:
        final = True; reasons.append('fusion_sensor+social')
    return {'alert': final, 'reasons': reasons, 'sensor_score': sensor_score, 'tweet_alerts': tweet_alerts, 'avg_tweet_conf': avg_conf}

if __name__=='__main__':
    # demo: sensor window is list of dicts (use your demo_data/sensor_demo.csv rows)
    sensor_window = [{'vibration':10,'flame':0,'water':30} for _ in range(10)]
    tweets = ["Flood reported near river bank", "We need help!", "Beautiful day"]
    print(fuse_and_decide(sensor_window, tweets))
