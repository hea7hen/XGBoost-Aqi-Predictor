"""
Quick training script to generate model files for deployment
This trains both Random Forest and XGBoost models and saves the best one
"""

import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import joblib

# XGBoost is optional - we'll use Random Forest which works reliably
HAS_XGB = False  # Set to False to avoid XGBoost dependency issues

def train_models(csv_path="aqi_with_binary.csv", use_xgb=False):
    """
    Train models and save the best one
    """
    # Check if CSV exists
    if not os.path.exists(csv_path):
        print(f"❌ Error: CSV file not found at: {csv_path}")
        print(f"\nPlease provide the path to your 'aqi_with_binary.csv' file.")
        print(f"Current working directory: {os.getcwd()}")
        return False
    
    print(f"📂 Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"✅ Data loaded. Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}\n")
    
    # Prepare data
    X = df.drop(columns=["Date", "AQI", "AQI_Binary"], errors="ignore")
    y = df["AQI"]
    
    print(f"Features ({len(X.columns)}): {list(X.columns)}\n")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("🔧 Training Random Forest Regressor...")
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_r2 = r2_score(y_test, rf_pred)
    rf_mae = mean_absolute_error(y_test, rf_pred)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
    
    print(f"  R² Score: {rf_r2:.4f}")
    print(f"  MAE: {rf_mae:.4f}")
    print(f"  RMSE: {rf_rmse:.4f}\n")
    
    xgb_r2 = 0
    xgb_pred = None
    xgb = None
    
    if use_xgb:
        try:
            print("🔧 Training XGBoost Regressor...")
            xgb = XGBRegressor(
                n_estimators=350,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                n_jobs=-1
            )
            xgb.fit(X_train, y_train)
            xgb_pred = xgb.predict(X_test)
            xgb_r2 = r2_score(y_test, xgb_pred)
            xgb_mae = mean_absolute_error(y_test, xgb_pred)
            xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
            
            print(f"  R² Score: {xgb_r2:.4f}")
            print(f"  MAE: {xgb_mae:.4f}")
            print(f"  RMSE: {xgb_rmse:.4f}\n")
        except Exception as e:
            print(f"⚠️  XGBoost training failed: {str(e)}")
            print("   Continuing with Random Forest only...\n")
            use_xgb = False
    
    # Compare models
    print("📊 Model Comparison:")
    print(f"  Random Forest R²: {rf_r2:.4f}")
    if use_xgb and xgb is not None:
        print(f"  XGBoost R²:      {xgb_r2:.4f}\n")
        
        # Save best model
        if xgb_r2 > rf_r2:
            best_model = xgb
            best_name = "XGBoost"
            joblib.dump(xgb, "best_aqi_model.joblib")
            print(f"✅ Saved XGBoost model to: best_aqi_model.joblib")
        else:
            best_model = rf
            best_name = "Random Forest"
            print(f"✅ Saved Random Forest model to: best_aqi_model.joblib")
            joblib.dump(rf, "best_aqi_model.joblib")
    else:
        print("  (XGBoost not available)\n")
        best_model = rf
        best_name = "Random Forest"
    
    # Always save Random Forest - save with both names for compatibility
    joblib.dump(rf, "rf_aqi_best_params_model.joblib")
    joblib.dump(rf, "best_aqi_model.joblib")  # Also save as best_aqi_model for Streamlit
    print(f"✅ Saved Random Forest model to: rf_aqi_best_params_model.joblib")
    print(f"✅ Saved Random Forest model to: best_aqi_model.joblib")
    
    print(f"\n🏆 Best model: {best_name} (R² = {max(rf_r2, xgb_r2) if xgb_r2 > 0 else rf_r2:.4f})")
    print(f"\n✨ Models are ready! You can now run the Streamlit app:")
    print(f"   python3 -m streamlit run streamlit_app.py")
    
    return True

if __name__ == "__main__":
    # Allow CSV path as command line argument
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "aqi_with_binary.csv"
    
    # Try multiple possible locations
    possible_paths = [
        csv_path,
        os.path.join(os.getcwd(), csv_path),
        os.path.join(os.path.expanduser("~"), "Downloads", csv_path),
        os.path.join(os.path.expanduser("~"), "Desktop", csv_path),
    ]
    
    csv_found = None
    for path in possible_paths:
        if os.path.exists(path):
            csv_found = path
            break
    
    if csv_found:
        print(f"📂 Found CSV at: {csv_found}\n")
        train_models(csv_found, use_xgb=False)  # Using Random Forest only for reliability
    else:
        print("❌ CSV file not found in common locations.")
        print("\nPlease provide the full path to 'aqi_with_binary.csv':")
        print("   python3 train_and_save_model.py /path/to/aqi_with_binary.csv")
        print("\nOr place the CSV file in the current directory.")

