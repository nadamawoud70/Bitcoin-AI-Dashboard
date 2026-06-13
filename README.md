# 🚀 Bitcoin AI Price Forecasting Dashboard

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![Machine Learning](https://img.shields.io/badge/Machine_Learning-Scikit_Learn-orange)

## 📌 Project Overview
This project is an end-to-end AI and Data Engineering application that tracks real-time Bitcoin prices and predicts future price movements using Machine Learning. 

The system operates via an automated pipeline that fetches live cryptocurrency market data, processes it, engineers time-series features, trains a predictive model, and serves the insights through an interactive and user-friendly Streamlit dashboard.

---

## ✨ Features
* **Real-Time Data Collection:** Automated fetching of live Bitcoin market data via the CoinGecko API.
* **Automated Data Pipeline:** Cleans, preprocesses, and structures the collected data seamlessly.
* **Time-Series Feature Engineering:** Generates crucial lag features (`Price_Lag_1`, `Price_Lag_2`) to improve forecasting accuracy.
* **Machine Learning Forecasting:** Utilizes a trained Linear Regression model to predict the next Bitcoin price.
* **Interactive Dashboard:** A Streamlit-based UI displaying:
  * Current Bitcoin Price & Price Change
  * AI-Predicted Price
  * Historical Price Trends (Interactive Charts)
  * Latest Pipeline Data logs
* **Modular Architecture:** Clean, maintainable, and GitHub-ready project structure.

---

## ⚙️ Workflow Architecture

1. **Fetch:** Pull real-time Bitcoin price data from the CoinGecko API.
2. **Preprocess:** Clean data and generate lag features for forecasting.
3. **Store:** Save the processed dataset locally in the `data/` directory.
4. **Train:** Train a Linear Regression model on the processed data.
5. **Serialize:** Save the trained model as a reusable artifact in the `models/` directory using `joblib`.
6. **Deploy:** Load the dataset and model into a Streamlit application to visualize real-time insights.

---

## 🛠️ Technologies Used

* **Language:** Python
* **Data Processing & Math:** Pandas, NumPy
* **API Integration:** Requests
* **Machine Learning:** Scikit-learn, Joblib
* **Web Application:** Streamlit
* **Version Control:** Git & GitHub

---

## 📁 Project Structure

```text
Bitcoin_AI_Project/
│
├── app.py                 # Streamlit Dashboard
├── pipeline.py            # Data Pipeline + ML Training
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── .gitignore             # Git ignore rules
├── run.sh                 # Run script
│
├── data/
│     └── crypto_data.csv  # Processed data
│
├── models/
│     └── crypto_model.pkl # Trained model
│
├── logs/                  # Logs
│
└── .venv/                 # Virtual environment (not pushed to GitHub)
