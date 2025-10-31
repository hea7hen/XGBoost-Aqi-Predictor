# 🚀 Quick Start Guide

## Running the Streamlit App

If you get `command not found: streamlit`, use one of these methods:

### Method 1: Using python3 -m (Recommended)
```bash
python3 -m streamlit run streamlit_app.py
```

### Method 2: Using pip3's executable path
```bash
python3 -m pip show streamlit | grep Location
# Then add that location/bin to your PATH, or use:
~/.local/bin/streamlit run streamlit_app.py
```

### Method 3: Create an alias (add to ~/.zshrc)
```bash
alias streamlit='python3 -m streamlit'
```

Then reload your shell:
```bash
source ~/.zshrc
```

## Running the Flask API

```bash
python3 app.py
```

The API will be available at `http://localhost:5000`

## Checking Your Installation

```bash
# Check if streamlit is installed
python3 -m streamlit --version

# Check if flask is installed
python3 -c "import flask; print(flask.__version__)"
```

## Important: Model Files Required

Before running the apps, make sure you have:
1. ✅ Trained and saved your model: `best_aqi_model.joblib` or `rf_aqi_best_params_model.joblib`
2. ✅ If using binary classification, run: `python3 save_models_for_deployment.py`

The apps will look for model files in the same directory.


