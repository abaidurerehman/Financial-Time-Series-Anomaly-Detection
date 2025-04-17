
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# --- Streamlit Page Config ---
st.set_page_config(page_title="Stock Price Anomaly Detection", layout="wide")

# --- Title ---
st.title("📈 Financial Time-Series Anomaly Detection")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload a Stock CSV file", type=['csv'])

if uploaded_file:
    # Read CSV
    df = pd.read_csv(uploaded_file)

    # --- Preprocessing ---
    df.columns = df.columns.str.strip()  # Remove spaces from column names
    df['Close/Last'] = df['Close/Last'].replace('[\$,]', '', regex=True).astype(float)  # Remove $ and convert to float
    df['Open'] = df['Open'].replace('[\$,]', '', regex=True).astype(float)
    df['High'] = df['High'].replace('[\$,]', '', regex=True).astype(float)
    df['Low'] = df['Low'].replace('[\$,]', '', regex=True).astype(float)

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    # --- Feature Engineering ---
    df['SMA_20'] = df['Close/Last'].rolling(window=20).mean()
    df['EMA_20'] = df['Close/Last'].ewm(span=20, adjust=False).mean()

    # RSI Calculation
    delta = df['Close/Last'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    RS = gain / loss
    df['RSI'] = 100 - (100 / (1 + RS))

    # Bollinger Bands
    df['BB_upper'] = df['SMA_20'] + 2 * df['Close/Last'].rolling(window=20).std()
    df['BB_lower'] = df['SMA_20'] - 2 * df['Close/Last'].rolling(window=20).std()

    # --- Anomaly Detection ---
    features = ['Close/Last', 'SMA_20', 'EMA_20', 'RSI']
    df_anomaly = df.dropna(subset=features)  # Drop rows with NaN
    scaler = StandardScaler()
    X = scaler.fit_transform(df_anomaly[features])

    model = IsolationForest(contamination=0.02, random_state=42)
    df_anomaly['anomaly'] = model.fit_predict(X)
    df_anomaly['anomaly'] = df_anomaly['anomaly'].map({1: 0, -1: 1})  # 1 for anomaly

    # --- Visualization ---
    st.subheader("📊 Stock Price with Anomalies")

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(df_anomaly['Date'], df_anomaly['Close/Last'], label='Close Price')
    ax.scatter(df_anomaly[df_anomaly['anomaly'] == 1]['Date'],
               df_anomaly[df_anomaly['anomaly'] == 1]['Close/Last'],
               color='red', label='Anomaly', marker='x', s=100)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.set_title('Stock Price with Detected Anomalies')
    ax.legend()
    st.pyplot(fig)

    st.subheader("📄 Raw Data with Anomaly Labels")
    st.dataframe(df_anomaly)

else:
    st.info("Please upload a CSV file to get started.")
