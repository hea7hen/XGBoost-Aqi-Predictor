"""
Simple test script for the Flask API
Run this to test your deployed API
"""

import requests
import json

BASE_URL = "http://localhost:5000"

def test_health():
    """Test health endpoint"""
    print("Testing /health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_predict():
    """Test prediction endpoint"""
    print("Testing /predict endpoint...")
    
    sample_data = {
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
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json=sample_data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_batch():
    """Test batch prediction endpoint"""
    print("Testing /predict/batch endpoint...")
    
    samples = [
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
        },
        {
            "PM2.5": 150,
            "PM10": 200,
            "NO": 30,
            "NO2": 50,
            "NOx": 80,
            "NH3": 25,
            "CO": 1.5,
            "SO2": 15,
            "O3": 100,
            "Benzene": 3.0,
            "Toluene": 5.0,
            "Xylene": 3.5
        }
    ]
    
    response = requests.post(
        f"{BASE_URL}/predict/batch",
        json={"samples": samples},
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

if __name__ == "__main__":
    print("🚀 Testing AQI Prediction API\n")
    print("=" * 50)
    
    try:
        test_health()
        test_predict()
        test_batch()
        print("✅ All tests completed!")
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed. Make sure the API is running:")
        print("   Run: python app.py")
    except Exception as e:
        print(f"❌ Error: {str(e)}")


