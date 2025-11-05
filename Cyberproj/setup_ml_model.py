# setup_ml_model.py

import json
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression


def create_and_save_model():
    print("Creating model features + dummy logistic model...")

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

    with open("model_features.json", "w") as f:
        json.dump(model_features, f)

    print("Created model_features.json")

    num_features = len(model_features)

    # Dummy dataset
    X = np.random.randint(0, 2, (50, num_features))
    y = (X[:, 2] | X[:, 9]).astype(int)    # make SMS/API calls "malicious"

    clf = LogisticRegression()
    clf.fit(X, y)

    joblib.dump(clf, "malware_model.joblib")
    print("Created malware_model.joblib")
    print("✅ Setup complete.")


if __name__ == "__main__":
    create_and_save_model()
