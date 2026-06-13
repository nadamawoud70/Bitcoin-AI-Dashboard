import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Bitcoin AI Dashboard",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 Real-Time Bitcoin Price Tracking")
st.subheader("AI & Data Engineering Dashboard")

DATA_PATH = "data/crypto_data.csv"
MODEL_PATH = "models/crypto_model.pkl"

if not os.path.exists(DATA_PATH):
    st.error("Data file not found. Run pipeline.py first.")
    st.stop()

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Run pipeline.py first.")
    st.stop()

df = pd.read_csv(DATA_PATH)

model = joblib.load(MODEL_PATH)

df["timestamp"] = pd.to_datetime(df["timestamp"])

current_price = df.iloc[-1]["Price"]
previous_price = df.iloc[-2]["Price"]

delta = current_price - previous_price

X = df[["Price_Lag_1", "Price_Lag_2"]].iloc[-1:]

prediction = model.predict(X)[0]

col1, col2, col3 = st.columns(3)

col1.metric(
    "Current Price",
    f"${current_price:,.2f}"
)

col2.metric(
    "Price Change",
    f"${delta:,.2f}"
)

col3.metric(
    "AI Prediction",
    f"${prediction:,.2f}"
)

st.markdown("---")

st.subheader("Bitcoin Price Trend")

chart = df.set_index("timestamp")[["Price"]]

st.line_chart(chart)

st.markdown("---")

st.subheader("Latest Pipeline Data")

st.dataframe(
    df.tail(10),
    use_container_width=True
)