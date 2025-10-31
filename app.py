"""
Flask REST API for AQI Prediction Model Deployment
Supports both AQI regression and binary classification (Safe/Dangerous)
"""

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

app = Flask(__name__)

# Model paths (update these based on your saved models)
MODEL_PATH_RF = "rf_aqi_best_params_model.joblib"
MODEL_PATH_XGB = "best_aqi_model.joblib"
MODEL_PATH_LR = "lr_aqi_classifier.joblib"  # You may need to save this separately

# Load models (with error handling)
models = {}
scaler = None

def load_models():
    """Load all available models"""
    global models, scaler
    
    # Load regression models
    if os.path.exists(MODEL_PATH_XGB):
        models['xgb'] = joblib.load(MODEL_PATH_XGB)
        print(f"✅ Loaded XGBoost model from {MODEL_PATH_XGB}")
    elif os.path.exists(MODEL_PATH_RF):
        models['rf'] = joblib.load(MODEL_PATH_RF)
        print(f"✅ Loaded Random Forest model from {MODEL_PATH_RF}")
    
    # Try to load scaler if it exists (for models that need it)
    global scaler
    scaler_path = "scaler.joblib"
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print(f"✅ Loaded scaler from {scaler_path}")
    
    # Load logistic regression models if they exist
    if os.path.exists(MODEL_PATH_LR):
        models['lr'] = joblib.load(MODEL_PATH_LR)
        print(f"✅ Loaded Logistic Regression from {MODEL_PATH_LR}")
    
    lr_simple_path = "lr_simple_aqi.joblib"
    if os.path.exists(lr_simple_path):
        models['lr_simple'] = joblib.load(lr_simple_path)
        print(f"✅ Loaded Simple Logistic Regression from {lr_simple_path}")

# Load models on startup
load_models()

# Expected features (based on your code - update if needed)
# These are all columns except Date, AQI, AQI_Binary
EXPECTED_FEATURES = [
    'PM10', 'PM2.5', 'SO2', 'NO2', 'CO', 'O3', 'NH3', 'Benzene',
    'Temp', 'RH', 'WS', 'WD', 'BP'
]

def validate_input(data):
    """Validate input data has all required features"""
    if isinstance(data, dict):
        missing = [f for f in EXPECTED_FEATURES if f not in data]
        if missing:
            return False, f"Missing features: {missing}"
        return True, None
    return False, "Invalid input format"

@app.route('/', methods=['GET'])
def home():
    """API home endpoint"""
    return jsonify({
        "message": "AQI Prediction API",
        "endpoints": {
            "/predict": "POST - Predict AQI value",
            "/predict/binary": "POST - Classify as Safe/Dangerous",
            "/health": "GET - Health check"
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    models_status = {name: "loaded" for name in models.keys()}
    return jsonify({
        "status": "healthy",
        "models_loaded": models_status,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict_aqi():
    """
    Predict AQI value (regression)
    
    Expected JSON format:
    {
        "PM2.5": 80,
        "PM10": 120,
        "NO": 10,
        "NO2": 20,
        "NOx": 30,
        "NH3": 15,
        "CO": 0.5,
        "SO2": 5,
        "O3": 50,
        "Benzene": 1.0,
        "Toluene": 2.0,
        "Xylene": 1.5
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Validate input
        valid, error_msg = validate_input(data)
        if not valid:
            return jsonify({"error": error_msg}), 400
        
        # Prepare input as DataFrame
        input_data = pd.DataFrame([data])
        input_data = input_data[EXPECTED_FEATURES]  # Ensure correct order
        
        # Get model (prefer XGBoost, fallback to RF)
        model = models.get('xgb') or models.get('rf')
        if model is None:
            return jsonify({"error": "No regression model loaded"}), 500
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        # Determine safety category
        if prediction <= 50:
            category = "Good"
        elif prediction <= 100:
            category = "Moderate"
        elif prediction <= 150:
            category = "Unhealthy for Sensitive Groups"
        elif prediction <= 200:
            category = "Unhealthy"
        elif prediction <= 300:
            category = "Very Unhealthy"
        else:
            category = "Hazardous"
        
        return jsonify({
            "predicted_aqi": float(prediction),
            "category": category,
            "is_safe": prediction <= 100,
            "input_features": data,
            "model_used": "XGBoost" if 'xgb' in models else "Random Forest"
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict/binary', methods=['POST'])
def predict_binary():
    """
    Classify AQI as Safe (0) or Dangerous (1)
    
    Expected JSON format (same as /predict):
    {
        "PM2.5": 80,
        "PM10": 120,
        ...
    }
    
    Or simpler format with just AQI:
    {
        "AQI": 120
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # If AQI is provided directly, use simpler logistic regression
        if 'AQI' in data and len(data) == 1:
            # This uses the simple logistic regression model
            # You'll need to load and save this model first
            if 'lr_simple' not in models:
                return jsonify({"error": "Simple logistic regression model not loaded"}), 500
            
            aqi_value = data['AQI']
            prediction = models['lr_simple'].predict([[aqi_value]])[0]
            outcome = "Safe" if prediction == 0 else "Dangerous"
            
            return jsonify({
                "prediction": int(prediction),
                "outcome": outcome,
                "is_safe": prediction == 0,
                "aqi_input": aqi_value
            })
        
        # Otherwise use full feature set with multi-feature logistic regression
        valid, error_msg = validate_input(data)
        if not valid:
            return jsonify({"error": error_msg}), 400
        
        if 'lr' not in models and 'lr_full' not in models:
            return jsonify({"error": "Logistic regression model not loaded"}), 500
        
        # Prepare input
        input_data = pd.DataFrame([data])
        input_data = input_data[EXPECTED_FEATURES]
        
        # Use scaler if available
        if scaler:
            input_data_scaled = scaler.transform(input_data.values)
        else:
            input_data_scaled = input_data.values
        
        # Get model
        lr_model = models.get('lr') or models.get('lr_full')
        prediction = lr_model.predict(input_data_scaled)[0]
        probability = lr_model.predict_proba(input_data_scaled)[0]
        
        outcome = "Safe" if prediction == 0 else "Dangerous"
        
        return jsonify({
            "prediction": int(prediction),
            "outcome": outcome,
            "is_safe": prediction == 0,
            "confidence": {
                "safe": float(probability[0]),
                "dangerous": float(probability[1])
            },
            "input_features": data
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Predict AQI for multiple samples at once
    
    Expected JSON format:
    {
        "samples": [
            {"PM2.5": 80, "PM10": 120, ...},
            {"PM2.5": 90, "PM10": 130, ...}
        ]
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'samples' not in data:
            return jsonify({"error": "No samples provided. Expected format: {'samples': [...]}"}), 400
        
        samples = data['samples']
        if not isinstance(samples, list):
            return jsonify({"error": "samples must be a list"}), 400
        
        # Validate all samples
        for i, sample in enumerate(samples):
            valid, error_msg = validate_input(sample)
            if not valid:
                return jsonify({"error": f"Sample {i}: {error_msg}"}), 400
        
        # Prepare DataFrame
        input_df = pd.DataFrame(samples)
        input_df = input_df[EXPECTED_FEATURES]
        
        # Get model
        model = models.get('xgb') or models.get('rf')
        if model is None:
            return jsonify({"error": "No regression model loaded"}), 500
        
        # Make predictions
        predictions = model.predict(input_df)
        
        results = []
        for i, pred in enumerate(predictions):
            category = "Good" if pred <= 50 else "Moderate" if pred <= 100 else "Unhealthy for Sensitive Groups" if pred <= 150 else "Unhealthy" if pred <= 200 else "Very Unhealthy" if pred <= 300 else "Hazardous"
            results.append({
                "sample_index": i,
                "predicted_aqi": float(pred),
                "category": category,
                "is_safe": pred <= 100
            })
        
        return jsonify({
            "predictions": results,
            "total_samples": len(samples),
            "model_used": "XGBoost" if 'xgb' in models else "Random Forest"
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting AQI Prediction API Server...")
    print("Available models:", list(models.keys()))
    print("📖 API Documentation:")
    print("  - GET  / : API information")
    print("  - GET  /health : Health check")
    print("  - POST /predict : Predict AQI value")
    print("  - POST /predict/binary : Classify Safe/Dangerous")
    print("  - POST /predict/batch : Batch predictions")
    
    app.run(host='0.0.0.0', port=5000, debug=True)

