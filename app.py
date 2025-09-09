from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import xgboost as xgb
from sklearn.metrics import accuracy_score
import warnings

# Suppress warnings for a cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# Initialize the Flask application
app = Flask(__name__)
# Enable CORS to allow cross-origin requests
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
    data.dropna(inplace=True)
    
    # 3. Target Variable
    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    
    # Isolate data for training and the last day for prediction
    df_train = data.iloc[:-1]
    last_day_features = data.iloc[-1:]

    features = [col for col in df_train.columns if col not in ['Target', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']]
    X = df_train[features]
    y = df_train['Target']
    
    # 4. Train the Model (on all available data for best prediction)
    model = xgb.XGBClassifier(
        objective='binary:logistic', n_estimators=100, learning_rate=0.1,
        use_label_encoder=False, eval_metric='logloss'
    )
    model.fit(X, y)
    
    # Evaluate on the full dataset to give a general idea of performance
    y_pred_all = model.predict(X)
    accuracy = accuracy_score(y, y_pred_all)

    # 5. Predict for the next day
    prediction_features = last_day_features[features]
    next_day_prediction = model.predict(prediction_features)[0]
    prediction_text = "UP" if next_day_prediction == 1 else "DOWN"

    return {
        "prediction_text": prediction_text,
        "accuracy": f"{accuracy:.2%}",
        "ticker": ticker
    }

# --- Flask Routes ---

@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint to handle prediction requests."""
    req_data = request.get_json()
    ticker = req_data.get('ticker')
    if not ticker:
        return jsonify({"error": "Ticker symbol is required."}), 400
    
    results = get_prediction_data(ticker.upper())
    return jsonify(results)

# This block is for local testing (optional)
if __name__ == '__main__':
    app.run(debug=True)

