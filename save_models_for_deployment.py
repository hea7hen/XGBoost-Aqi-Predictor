"""
Helper script to save models properly for deployment
Run this after training to ensure all necessary models and scalers are saved
"""

import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def save_all_models():
    """Save all models needed for deployment"""
    
    print("Loading data...")
    df = pd.read_csv("aqi_with_binary.csv")
    
    # Prepare features for regression (AQI prediction)
    X_reg = df.drop(columns=["Date", "AQI", "AQI_Binary"], errors="ignore")
    y_reg = df["AQI"]
    
    # Prepare features for binary classification
    X_clf = df.drop(columns=['Date', 'AQI', 'AQI_Binary'], errors="ignore")
    y_clf = df['AQI_Binary']
    
    print(f"Data shape: {X_reg.shape}")
    print(f"Features: {list(X_reg.columns)}\n")
    
    # 1. Save scaler for logistic regression
    print("Saving scaler...")
    scaler = StandardScaler()
    scaler.fit(X_clf)
    joblib.dump(scaler, "scaler.joblib")
    print("✅ Scaler saved to: scaler.joblib\n")
    
    # 2. Save logistic regression model for binary classification
    print("Training and saving Logistic Regression model...")
    X_scaled = scaler.transform(X_clf)
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_scaled, y_clf)
    joblib.dump(lr_model, "lr_aqi_classifier.joblib")
    print("✅ Logistic Regression saved to: lr_aqi_classifier.joblib\n")
    
    # 3. Save simple logistic regression (uses only AQI)
    print("Training and saving simple Logistic Regression (AQI only)...")
    X_simple = df[['AQI']]
    y_simple = df['AQI_Binary']
    lr_simple = LogisticRegression(max_iter=1000, random_state=42)
    lr_simple.fit(X_simple, y_simple)
    joblib.dump(lr_simple, "lr_simple_aqi.joblib")
    print("✅ Simple Logistic Regression saved to: lr_simple_aqi.joblib\n")
    
    print("📝 Note: Regression models (Random Forest/XGBoost) should be saved")
    print("   from your main training script (cie3_sl.py)")
    print("   Expected files: best_aqi_model.joblib or rf_aqi_best_params_model.joblib\n")
    
    print("✅ All classification models saved!")
    print("\nNext steps:")
    print("1. Ensure your regression model is saved (run your training script)")
    print("2. Copy all .joblib files to your deployment directory")
    print("3. Run: streamlit run streamlit_app.py")
    print("   OR: python app.py")

if __name__ == "__main__":
    save_all_models()


