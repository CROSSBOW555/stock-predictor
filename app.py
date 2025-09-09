from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import xgboost as xgb
from sklearn.metrics import accuracy_score
import warnings
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend for Matplotlib

# --- Flask App Initialization ---
# The key change is here: template_folder='.' tells Flask to look for index.html
# in the same directory as this app.py file.
app = Flask(__name__, template_folder='.')
CORS(app)

# --- Data Processing and Model Logic ---
def get_prediction_data(ticker):
    """The main logic function to get data, train model, and predict."""
    
    # 1. Download data
    data = yf.Ticker(ticker).history(start='2015-01-01', end='2025-12-31')
    if data.empty:
        return {"error": f"Invalid ticker '{ticker}' or no data available."}

    # 2. Feature Engineering
    data.ta.sma(length=20, append=True)
    data.ta.ema(length=20, append=True)
    data.ta.rsi(length=14, append=True)
    data.ta.macd(fast=12, slow=26, signal=9, append=True)
    
    # Create the target variable before dropping NaN values
    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    data.dropna(inplace=True)
    
    # Ensure there's enough data after processing
    if len(data) < 252:
        return {"error": f"Not enough historical data for '{ticker}' to make a reliable prediction."}

    features = [col for col in data.columns if col not in ['Target', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']]
    X = data[features]
    y = data['Target']
    
    # Split into a training set and a test set (last year)
    train_size = len(data) - 252
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    # 3. Train the Model
    model = xgb.XGBClassifier(
        objective='binary:logistic', n_estimators=100, learning_rate=0.1,
        use_label_encoder=False, eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    
    # 4. Evaluate the Model on the test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # 5. Predict for the next day using the very last row of data
    last_day_features = data.iloc[-1:][features]
    next_day_prediction = model.predict(last_day_features)[0]
    prediction_text = "UP" if next_day_prediction == 1 else "DOWN"
    
    return {
        "prediction_text": prediction_text,
        "accuracy": f"{accuracy:.2%}",
        "ticker": ticker.upper()
    }

# --- API Endpoints ---
@app.route('/')
def home():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the prediction request from the frontend."""
    req_data = request.get_json()
    ticker = req_data.get('ticker')
    
    if not ticker:
        return jsonify({"error": "Ticker symbol is required."}), 400
    
    results = get_prediction_data(ticker)
    return jsonify(results)

# This part is for local testing and not used by Gunicorn on Render
if __name__ == "__main__":
    app.run(debug=True)

