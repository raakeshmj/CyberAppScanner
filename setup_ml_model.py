# setup_ml_model.py

import json
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

def create_and_save_model():
    """
    Creates a dummy ML model and the feature list, then saves them to disk.
    This simulates having a real, pre-trained model.
    """
    print("Creating placeholder ML model and feature list...")

    # 1. Define the features the model expects.
    # In a real scenario, this list would be generated from a massive dataset.
    model_features = [
        "permission:android.permission.INTERNET",
        "permission:android.permission.ACCESS_NETWORK_STATE",
        "permission:android.permission.SEND_SMS",
        "permission:android.permission.RECEIVE_SMS",
        "permission:android.permission.READ_SMS",
        "permission:android.permission.READ_CONTACTS",
        "permission:android.permission.WRITE_EXTERNAL_STORAGE",
        "permission:android.permission.CAMERA",
        "permission:android.permission.ACCESS_FINE_LOCATION",
        "api:Landroid/telephony/SmsManager;->sendTextMessage",
        "api:Ljava/net/URL;->openConnection",
        "api:Ljava/lang/reflect/Method;->invoke",
        "api:Landroid/content/ContentResolver;->query"
    ]

    # Save the feature list to a JSON file
    with open('model_features.json', 'w') as f:
        json.dump(model_features, f)
    print(" -> 'model_features.json' created successfully.")

    # 2. Create and train a dummy Logistic Regression model.
    # We create a simple dataset with the correct number of features.
    num_features = len(model_features)
    # Dummy data: 2 "benign" samples, 2 "malicious" samples
    X_dummy = np.array([
        np.zeros(num_features),                 # A sample with no features
        np.random.randint(0, 2, num_features),  # A random benign-like sample
        np.random.randint(0, 2, num_features),  # A random malicious-like sample
        np.ones(num_features)                   # A sample with all features
    ])
    # Dummy labels: 0 for benign, 1 for malicious
    y_dummy = np.array([0, 0, 1, 1])

    # Create and "train" the model
    model = LogisticRegression()
    model.fit(X_dummy, y_dummy)
    print(" -> Dummy model trained.")

    # 3. Save the trained model to a file
    joblib.dump(model, 'malware_model.joblib')
    print(" -> 'malware_model.joblib' created successfully.")
    print("\nSetup complete. You can now run the dashboard.")

if __name__ == "__main__":
    create_and_save_model()