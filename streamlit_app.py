"""
Streamlit Web App for AQI Prediction Model Deployment
Interactive UI for air quality predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(
    page_title="AQI Prediction Dashboard",
    page_icon="🌍",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Model paths
MODEL_PATH_RF = "rf_aqi_best_params_model.joblib"
MODEL_PATH_XGB = "best_aqi_model.joblib"

@st.cache_resource
def load_model():
    """Load the best available model"""
    if os.path.exists(MODEL_PATH_XGB):
        return joblib.load(MODEL_PATH_XGB), "XGBoost"
    elif os.path.exists(MODEL_PATH_RF):
        return joblib.load(MODEL_PATH_RF), "Random Forest"
    else:
        return None, None

def get_aqi_category(aqi):
    """Get AQI category and color"""
    if aqi <= 50:
        return "Good", "🟢", "#00e400"
    elif aqi <= 100:
        return "Moderate", "🟡", "#ffff00"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "🟠", "#ff7e00"
    elif aqi <= 200:
        return "Unhealthy", "🔴", "#ff0000"
    elif aqi <= 300:
        return "Very Unhealthy", "🟣", "#8f3f97"
    else:
        return "Hazardous", "⚫", "#7e0023"

def main():
    # Header
    st.markdown('<p class="main-header">🌍 Air Quality Index (AQI) Prediction Dashboard</p>', unsafe_allow_html=True)
    
    # Load model
    model, model_name = load_model()
    
    if model is None:
        st.error("❌ No model found! Please ensure you have trained and saved a model.")
        st.info("Run your training script first to generate model files (e.g., `best_aqi_model.joblib`)")
        return
    
    st.sidebar.success(f"✅ Model loaded: {model_name}")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["🔮 Predict AQI", "📊 Batch Prediction", "ℹ️ About"])
    
    # Tab 1: Single Prediction
    with tab1:
        st.header("Predict AQI from Air Quality Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Air Quality Parameters")
            pm10 = st.number_input("PM10 (μg/m³)", min_value=0.0, max_value=500.0, value=120.0, step=1.0)
            pm25 = st.number_input("PM2.5 (μg/m³)", min_value=0.0, max_value=500.0, value=80.0, step=1.0)
            so2 = st.number_input("SO2 (μg/m³)", min_value=0.0, max_value=100.0, value=5.0, step=0.1)
            no2 = st.number_input("NO2 (μg/m³)", min_value=0.0, max_value=100.0, value=20.0, step=0.1)
            co = st.number_input("CO (mg/m³)", min_value=0.0, max_value=10.0, value=0.5, step=0.01)
            o3 = st.number_input("O3 (μg/m³)", min_value=0.0, max_value=200.0, value=50.0, step=0.1)
            nh3 = st.number_input("NH3 (μg/m³)", min_value=0.0, max_value=200.0, value=15.0, step=0.1)
        
        with col2:
            st.subheader("Additional Parameters")
            benzene = st.number_input("Benzene (μg/m³)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
            temp = st.number_input("Temperature (°C)", min_value=-20.0, max_value=50.0, value=25.0, step=0.1)
            rh = st.number_input("Relative Humidity (%)", min_value=0.0, max_value=100.0, value=60.0, step=1.0)
            ws = st.number_input("Wind Speed (m/s)", min_value=0.0, max_value=50.0, value=5.0, step=0.1)
            wd = st.number_input("Wind Direction (degrees)", min_value=0.0, max_value=360.0, value=180.0, step=1.0)
            bp = st.number_input("Barometric Pressure (hPa)", min_value=900.0, max_value=1100.0, value=1013.0, step=0.1)
        
        # Prepare input (in correct order)
        input_data = {
            'PM10': pm10,
            'PM2.5': pm25,
            'SO2': so2,
            'NO2': no2,
            'CO': co,
            'O3': o3,
            'NH3': nh3,
            'Benzene': benzene,
            'Temp': temp,
            'RH': rh,
            'WS': ws,
            'WD': wd,
            'BP': bp
        }
        
        if st.button("🔮 Predict AQI", type="primary", use_container_width=True):
            try:
                # Create DataFrame
                input_df = pd.DataFrame([input_data])
                
                # Ensure correct feature order (get from model if possible)
                feature_names = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else input_df.columns
                input_df = input_df[feature_names] if hasattr(model, 'feature_names_in_') else input_df
                
                # Predict
                prediction = model.predict(input_df)[0]
                
                # Get category
                category, emoji, color = get_aqi_category(prediction)
                
                # Display results
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Predicted AQI", f"{prediction:.1f}")
                
                with col2:
                    st.markdown(f"### {emoji} {category}")
                
                with col3:
                    is_safe = "✅ Safe" if prediction <= 100 else "⚠️ Unhealthy"
                    st.markdown(f"### {is_safe}")
                
                # Visual indicator
                st.markdown(f"""
                <div class="prediction-box">
                    <h3 style="color: {color};">Predicted AQI: {prediction:.1f}</h3>
                    <p><strong>Category:</strong> {emoji} {category}</p>
                    <p><strong>Status:</strong> {'Safe for most people' if prediction <= 100 else 'May cause health issues'}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Bar chart showing AQI levels
                categories = ["Good", "Moderate", "USG", "Unhealthy", "Very Unhealthy", "Hazardous"]
                thresholds = [0, 50, 100, 150, 200, 300, 500]
                fig, ax = plt.subplots(figsize=(10, 2))
                ax.barh([0], [prediction], color=color, alpha=0.7, height=0.5)
                for i, thresh in enumerate(thresholds[1:], 1):
                    ax.axvline(thresh, color='gray', linestyle='--', alpha=0.5)
                ax.set_xlim(0, max(500, prediction * 1.2))
                ax.set_xlabel("AQI Value")
                ax.set_title("AQI Prediction Scale")
                ax.set_yticks([])
                st.pyplot(fig)
                
                # Show input summary
                with st.expander("📋 View Input Parameters"):
                    st.json(input_data)
                
            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")
                st.exception(e)
    
    # Tab 2: Batch Prediction
    with tab2:
        st.header("Batch Prediction")
        st.info("Upload a CSV file with air quality parameters to predict AQI for multiple samples")
        
        uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.write("Preview of uploaded data:")
                st.dataframe(df.head())
                
                # Check required columns
                required_cols = ['PM10', 'PM2.5', 'SO2', 'NO2', 'CO', 'O3', 'NH3', 'Benzene', 'Temp', 'RH', 'WS', 'WD', 'BP']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    st.error(f"❌ Missing required columns: {missing_cols}")
                else:
                    if st.button("🔮 Predict for All Rows", type="primary"):
                        with st.spinner("Predicting..."):
                            feature_names = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else required_cols
                            input_df = df[feature_names] if hasattr(model, 'feature_names_in_') else df[required_cols]
                            predictions = model.predict(input_df)
                            
                            # Add predictions to dataframe
                            result_df = df.copy()
                            result_df['Predicted_AQI'] = predictions
                            result_df['Category'] = result_df['Predicted_AQI'].apply(lambda x: get_aqi_category(x)[0])
                            result_df['Is_Safe'] = result_df['Predicted_AQI'] <= 100
                            
                            st.success(f"✅ Predictions complete for {len(df)} samples!")
                            st.dataframe(result_df)
                            
                            # Download results
                            csv = result_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Predictions as CSV",
                                data=csv,
                                file_name="aqi_predictions.csv",
                                mime="text/csv"
                            )
                            
                            # Visualization
                            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                            
                            # Distribution
                            axes[0].hist(predictions, bins=20, edgecolor='k', alpha=0.7)
                            axes[0].set_xlabel("Predicted AQI")
                            axes[0].set_ylabel("Frequency")
                            axes[0].set_title("Distribution of Predicted AQI")
                            axes[0].grid(True, alpha=0.3)
                            
                            # Category counts
                            category_counts = result_df['Category'].value_counts()
                            axes[1].bar(range(len(category_counts)), category_counts.values)
                            axes[1].set_xticks(range(len(category_counts)))
                            axes[1].set_xticklabels(category_counts.index, rotation=45, ha='right')
                            axes[1].set_ylabel("Count")
                            axes[1].set_title("AQI Category Distribution")
                            axes[1].grid(True, alpha=0.3)
                            
                            plt.tight_layout()
                            st.pyplot(fig)
            
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
                st.exception(e)
    
    # Tab 3: About
    with tab3:
        st.header("About This Application")
        st.markdown("""
        ### 🌍 AQI Prediction Model
        
        This application uses machine learning models to predict Air Quality Index (AQI) based on various air quality parameters.
        
        **Features:**
        - Real-time AQI prediction from air quality parameters
        - Batch prediction support via CSV upload
        - Interactive visualization of results
        
        **Model Information:**
        - Model Type: Regression (Random Forest / XGBoost)
        - Target: Air Quality Index (AQI)
        
        **Required Parameters:**
        - PM2.5, PM10 (Particulate Matter)
        - SO2 (Sulfur Dioxide)
        - NO2 (Nitrogen Dioxide)
        - CO (Carbon Monoxide)
        - O3 (Ozone)
        - NH3 (Ammonia)
        - Benzene (Volatile Organic Compound)
        - Temp (Temperature)
        - RH (Relative Humidity)
        - WS (Wind Speed)
        - WD (Wind Direction)
        - BP (Barometric Pressure)
        
        **AQI Categories:**
        - 🟢 **Good (0-50)**: Air quality is satisfactory
        - 🟡 **Moderate (51-100)**: Air quality is acceptable
        - 🟠 **Unhealthy for Sensitive Groups (101-150)**: May affect sensitive individuals
        - 🔴 **Unhealthy (151-200)**: Everyone may begin to experience health effects
        - 🟣 **Very Unhealthy (201-300)**: Health alert - everyone may experience serious health effects
        - ⚫ **Hazardous (301+)**: Health warning - entire population likely affected
        
        **Instructions:**
        1. Enter air quality parameter values in the "Predict AQI" tab
        2. Click "Predict AQI" to get the prediction
        3. For batch predictions, upload a CSV file in the "Batch Prediction" tab
        """)

if __name__ == "__main__":
    main()

