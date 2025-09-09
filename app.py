from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
import io
import base64

# Use a non-interactive backend for Matplotlib, required for server environments
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


# --- Flask App Initialization ---
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
    
    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    data.dropna(inplace=True)
    
    if len(data) < 252:
        return {"error": f"Not enough historical data for '{ticker}' to make a reliable prediction."}

    features = [col for col in data.columns if col not in ['Target', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']]
    X = data[features]
    y = data['Target']
    
    train_size = len(data) - 252
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    # 3. Train the Model
    model = xgb.XGBClassifier(
        objective='binary:logistic', n_estimators=100, learning_rate=0.1,
        use_label_encoder=False, eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    
    # 4. Evaluate and Create Confusion Matrix
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # --- NEW: Generate Confusion Matrix Plot ---
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Predicted Down', 'Predicted Up'],
                yticklabels=['Actual Down', 'Actual Up'])
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Model Performance (Confusion Matrix)')
    
    # Save plot to a memory buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig) # Close the figure to free up memory
    buf.seek(0)
    
    # Encode the image to Base64 to send it over the web
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    confusion_matrix_url = f"data:image/png;base64,{img_base64}"
    # --- END OF NEW PLOT CODE ---

    # 5. Predict for the next day
    last_day_features = data.iloc[-1:][features]
    next_day_prediction = model.predict(last_day_features)[0]
    prediction_direction = "UP" if next_day_prediction == 1 else "DOWN"
    prediction_summary = f"Based on the analysis of historical data, the model predicts that the stock for {ticker.upper()} will go {prediction_direction} on the next trading day."

    return {
        "prediction_text": prediction_direction,
        "prediction_summary": prediction_summary,
        "accuracy": f"{accuracy:.2%}",
        "ticker": ticker.upper(),
        "confusion_matrix_img": confusion_matrix_url # Add the image data to the response
    }

# --- API Endpoints ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    req_data = request.get_json()
    ticker = req_data.get('ticker')
    if not ticker:
        return jsonify({"error": "Ticker symbol is required."}), 400
    results = get_prediction_data(ticker)
    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)

