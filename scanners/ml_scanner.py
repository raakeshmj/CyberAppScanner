# scanners/ml_scanner.py

import json
import joblib
import numpy as np

# --- Configuration ---
MODEL_PATH = 'malware_model.joblib'
FEATURES_PATH = 'model_features.json'

# --- Core ML Functions ---

def load_model_and_features():
    """Loads the pre-trained model and the feature list."""
    try:
        model = joblib.load(MODEL_PATH)
        with open(FEATURES_PATH, 'r') as f:
            features = json.load(f)
        return model, features
    except FileNotFoundError:
        print("[ML Scanner] Model or feature file not found. Please download them.")
        return None, None

def extract_features(analysis_obj, feature_list):
    """
    Extracts features from an APK analysis object and creates a numerical vector.
    """
    if not analysis_obj:
        return None

    # Get all permissions requested by the APK
    try:
        requested_permissions = set(analysis_obj.get_permissions())
    except Exception:
        requested_permissions = set()

    # Get all API calls made by the APK
    api_calls = set()
    try:
        for method in analysis_obj.get_methods():
            if method.is_external(): continue
            for _, call, _ in method.get_xref_to():
                api_calls.add(f"{call.class_name}->{call.name}")
    except Exception:
        api_calls = set()

    # Create the feature vector
    # This is a binary vector (1 if the feature is present, 0 otherwise)
    vector = []
    for feature in feature_list:
        if feature.startswith('permission:'):
            # Check for permission
            permission_name = feature.split('permission:', 1)[1]
            vector.append(1 if permission_name in requested_permissions else 0)
        elif feature.startswith('api:'):
            # Check for API call
            api_name = feature.split('api:', 1)[1]
            vector.append(1 if api_name in api_calls else 0)
        else:
            vector.append(0)
    
    return np.array(vector).reshape(1, -1) # Reshape for a single prediction

def predict(analysis_obj):
    """
    Loads the model, extracts features, and predicts if the APK is malicious.
    """
    model, features = load_model_and_features()
    if not model or not features:
        return {"error": "Model or features not loaded."}
    
    feature_vector = extract_features(analysis_obj, features)
    if feature_vector is None:
        return {"error": "Could not extract features from APK."}
    
    # Predict the probability of being malware (class 1)
    try:
        prediction_proba = model.predict_proba(feature_vector)[0]
        malware_probability = prediction_proba[1] # Probability of the "malicious" class
    except Exception as e:
        return {"error": f"Prediction failed: {e}"}

    # Determine the verdict based on the probability
    verdict = "Benign"
    if malware_probability > 0.8:
        verdict = "Malicious"
    elif malware_probability > 0.5:
        verdict = "Suspicious"

    return {
        "malware_probability": malware_probability,
        "verdict": verdict
    }