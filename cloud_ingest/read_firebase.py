import os

def fetch_sensor_data(db_ref, limit=100):
    """Fetch sensor data from Firebase Realtime Database"""
    try:
        data = db_ref.order_by_child('ts').limit_to_last(limit).get().val()
        return data if data else []
    except Exception as e:
        print(f"Error fetching sensor data: {e}")
        return []

def fetch_tweets(db_ref, limit=100):
    """Fetch tweets from Firebase Realtime Database"""
    try:
        data = db_ref.order_by_child('ts').limit_to_last(limit).get().val()
        return data if data else []
    except Exception as e:
        print(f"Error fetching tweets: {e}")
        return []
