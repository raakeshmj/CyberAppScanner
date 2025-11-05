# scanners/ml_scanner.py

import os
import json
import numpy as np

try:
    import tensorflow as tf
    TF_OK = True
except:
    TF_OK = False

ROOT = os.path.join(os.path.dirname(__file__), "..")
FEATURE_PATH = os.path.join(ROOT, "model_features.json")
TF_MODEL_PATH = os.path.join(ROOT, "models", "gnn_model.h5")
JOBLIB_MODEL_PATH = os.path.join(ROOT, "malware_model.joblib")


_features = None
_tf_model = None
_joblib = None


def load_features():
    global _features
    if _features is None:
        with open(FEATURE_PATH) as f:
            _features = json.load(f)
    return _features


def extract_vector(analysis_obj):
    features = load_features()

    requested = set(analysis_obj["permissions"]["requested"])
    apis = set(analysis_obj["api_calls"])

    vec = []
    for f in features:
        if f.startswith("permission:"):
            p = f.split("permission:")[1]
            vec.append(1 if p in requested else 0)
        else:
            a = f.split("api:")[1]
            vec.append(1 if a in apis else 0)

    return np.array(vec, dtype=np.float32).reshape(1, -1)


def load_gnn():
    global _tf_model
    if _tf_model is None:
        _tf_model = tf.keras.models.load_model(TF_MODEL_PATH, compile=False)
    return _tf_model


def load_joblib():
    global _joblib
    if _joblib is None:
        import joblib
        _joblib = joblib.load(JOBLIB_MODEL_PATH)
    return _joblib


def predict(analysis_obj):
    try:
        x = extract_vector(analysis_obj)
        A = np.ones((1, 1, 1), dtype=np.float32)

        if TF_OK and os.path.exists(TF_MODEL_PATH):
            model = load_gnn()
            logits = model.predict([x, A])
            probs = tf.nn.softmax(logits, -1).numpy()[0]
            mal = float(probs[1])
            source = "gnn"

        else:
            clf = load_joblib()
            mal = float(clf.predict_proba(x)[0][1])
            source = "joblib"

        verdict = "Malicious" if mal > 0.85 else "Suspicious" if mal > 0.5 else "Benign"

        return {
            "malware_probability": mal,
            "verdict": verdict,
            "source": source
        }

    except Exception as e:
        return {"error": str(e)}
