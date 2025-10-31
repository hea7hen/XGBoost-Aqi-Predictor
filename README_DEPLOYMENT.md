# 🚀 AQI Prediction Model Deployment Guide

This guide explains how to deploy your AQI prediction models from `cie3_sl.py`.

## 📋 Prerequisites

1. **Trained Models**: Ensure you have saved model files:
   - `best_aqi_model.joblib` (XGBoost) or `rf_aqi_best_params_model.joblib` (Random Forest)
   - Optional: `lr_aqi_classifier.joblib` for binary classification

2. **Data File**: `aqi_with_binary.csv` (only needed for training, not deployment)

3. **Python 3.8+** installed

## 🔧 Installation

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## 🌐 Deployment Options

### Option 1: Streamlit Web App (Recommended for Quick Deployment)

The easiest way to deploy with an interactive UI.

**Run the app:**
```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

**Features:**
- Interactive input form for air quality parameters
- Real-time AQI prediction
- Batch prediction via CSV upload
- Visualizations and results download

**To deploy online:**
1. Push your code to GitHub
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your repository
4. Deploy!

### Option 2: Flask REST API (For Production/Integration)

For integrating with other applications or serving via REST API.

**Run the API:**
```bash
python app.py
```

The API will start at `http://localhost:5000`

**API Endpoints:**

1. **Health Check**
   ```bash
   GET http://localhost:5000/health
   ```

2. **Predict AQI (Regression)**
   ```bash
   POST http://localhost:5000/predict
   Content-Type: application/json
   
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
   ```

3. **Predict Binary (Safe/Dangerous)**
   ```bash
   POST http://localhost:5000/predict/binary
   Content-Type: application/json
   
   {
     "PM2.5": 80,
     "PM10": 120,
     ...
   }
   ```

4. **Batch Prediction**
   ```bash
   POST http://localhost:5000/predict/batch
   Content-Type: application/json
   
   {
     "samples": [
       {"PM2.5": 80, "PM10": 120, ...},
       {"PM2.5": 90, "PM10": 130, ...}
     ]
   }
   ```

**Example using curl:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

**Example using Python:**
```python
import requests

response = requests.post('http://localhost:5000/predict', json={
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
})

print(response.json())
```

## 📦 Preparing Your Models

Before deployment, ensure you've saved your models. Add this to the end of your `cie3_sl.py`:

```python
# Save scaler if needed (for logistic regression)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)  # X is your feature data
joblib.dump(scaler, "scaler.joblib")

# Save logistic regression model (if you want binary classification)
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_scaled, y_binary)  # Train on your data
joblib.dump(lr_model, "lr_aqi_classifier.joblib")
```

## 🌍 Production Deployment

### Deploy Flask API to Cloud Platforms

**Heroku:**
1. Install Heroku CLI
2. Create `Procfile`: `web: gunicorn app:app`
3. Run: `heroku create your-app-name && git push heroku main`

**AWS/GCP/Azure:**
- Use container services (Docker + Cloud Run/ECS/Lambda)
- Use serverless functions (AWS Lambda with API Gateway)

**Docker Deployment:**

Create `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["python", "app.py"]
```

Build and run:
```bash
docker build -t aqi-api .
docker run -p 5000:5000 aqi-api
```

## 🔒 Security Considerations

For production:
1. Add authentication/API keys
2. Rate limiting
3. Input validation (already included)
4. HTTPS/SSL
5. Environment variables for sensitive data

Example with API key (add to `app.py`):
```python
API_KEY = os.getenv('API_KEY', 'your-secret-key')

@app.before_request
def check_api_key():
    if request.endpoint != 'health' and request.endpoint != 'home':
        key = request.headers.get('X-API-Key')
        if key != API_KEY:
            return jsonify({"error": "Invalid API key"}), 401
```

## 📊 Testing Your Deployment

1. **Test Streamlit:**
   - Open the app
   - Enter sample values
   - Verify predictions appear

2. **Test Flask API:**
   ```bash
   # Health check
   curl http://localhost:5000/health
   
   # Prediction
   curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d @test_input.json
   ```

## 🐛 Troubleshooting

**Model not found error:**
- Ensure model files are in the same directory as the app
- Check file paths in `app.py` or `streamlit_app.py`

**Feature mismatch:**
- Verify your model was trained with the same features
- Update `EXPECTED_FEATURES` in `app.py` if needed

**Import errors:**
- Run `pip install -r requirements.txt`
- Check Python version (3.8+ required)

## 📝 Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Deploying ML Models](https://www.mlflow.org/docs/latest/models.html)

## 🤝 Support

If you encounter issues:
1. Check model files exist and are loadable
2. Verify input format matches expected features
3. Check error logs for specific error messages

---

**Happy Deploying! 🌍✨**


