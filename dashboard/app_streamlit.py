# dashboard/app_streamlit.py
import streamlit as st
import pandas as pd, requests, os
st.set_page_config(layout='wide', page_title='Disaster AI Dashboard')

st.title("Disaster Management with AI — Demo")
col1, col2 = st.columns(2)

# left: sensor chart
with col1:
    st.subheader("Sensor data (demo)")
    df = pd.read_csv(os.path.join('..','demo_data','sensor_demo.csv'))
    st.line_chart(df[['vibration','water']])
    st.write("Latest rows:")
    st.dataframe(df.tail(10))

# right: tweets & prediction
with col2:
    st.subheader("Social posts (demo)")
    tweets = pd.read_csv(os.path.join('..','demo_data','tweets_demo.csv'))
    st.dataframe(tweets.head(20))
    idx = st.number_input("Pick tweet row to classify", min_value=0, max_value=len(tweets)-1, value=0)
    tweet = tweets.loc[int(idx),'text']
    st.write("Tweet:", tweet)
    if st.button("Classify Tweet"):
        r = requests.post("http://localhost:5000/predict/text", json={'text': tweet}).json()
        if r.get('label')==1:
            st.error(f"Disaster-related (conf {r.get('confidence'):.2f})")
        else:
            st.success(f"Not disaster-related (conf {r.get('confidence'):.2f})")

st.markdown("---")
st.write("Use `orchestrator/orchestrator.py` to run fusion and simulate alerts.")
