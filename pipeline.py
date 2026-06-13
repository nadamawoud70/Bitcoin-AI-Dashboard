"""
====================================================
Bitcoin Real-Time Data Pipeline & ML Training
====================================================

1. Fetches the last 24 hours of Bitcoin prices
   from the CoinGecko API.

2. Cleans and preprocesses the data.

3. Creates lag features.

4. Saves processed data.

5. Trains a Linear Regression model.

6. Saves the trained model.

====================================================
"""

import os
import logging
import requests
import joblib
import pandas as pd

from sklearn.linear_model import LinearRegression


# -----------------------------
# Configuration
# -----------------------------

API_URL = (
    "https://api.coingecko.com/api/v3/coins/"
    "bitcoin/market_chart?vs_currency=usd&days=1"
)

DATA_DIR = "data"
MODEL_DIR = "models"

DATA_PATH = os.path.join(DATA_DIR, "crypto_data.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "crypto_model.pkl")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


# -----------------------------
# Create folders
# -----------------------------

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# -----------------------------
# Fetch data
# -----------------------------

def fetch_data():

    try:

        response = requests.get(
            API_URL,
            timeout=15
        )

        response.raise_for_status()

        prices = response.json()["prices"]

        df = pd.DataFrame(
            prices,
            columns=[
                "timestamp",
                "Price"
            ]
        )

        logging.info(
            "Data fetched successfully."
        )

        return df

    except Exception as e:

        logging.error(
            f"API Error: {e}"
        )

        return None


# -----------------------------
# Preprocessing
# -----------------------------

def preprocess(df):

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        unit="ms"
    )

    df = df.sort_values(
        "timestamp"
    )

    df["Price_Lag_1"] = df["Price"].shift(1)

    df["Price_Lag_2"] = df["Price"].shift(2)

    df = df.dropna()

    return df


# -----------------------------
# Train model
# -----------------------------

def train_model(df):

    X = df[
        [
            "Price_Lag_1",
            "Price_Lag_2"
        ]
    ]

    y = df["Price"]

    model = LinearRegression()

    model.fit(X, y)

    joblib.dump(
        model,
        MODEL_PATH
    )

    logging.info(
        f"Model saved -> {MODEL_PATH}"
    )


# -----------------------------
# Save data
# -----------------------------

def save_data(df):

    df.to_csv(
        DATA_PATH,
        index=False
    )

    logging.info(
        f"Data saved -> {DATA_PATH}"
    )


# -----------------------------
# Main
# -----------------------------

def main():

    logging.info(
        "Starting pipeline..."
    )

    df = fetch_data()

    if df is None:

        logging.error(
            "Pipeline stopped."
        )

        return

    df = preprocess(df)

    save_data(df)

    train_model(df)

    logging.info(
        "Pipeline completed successfully."
    )


if __name__ == "__main__":

    main()