from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from androguard.misc import AnalyzeAPK

# -------------------- FIXED GCN LAYER --------------------
class GCNLayer(tf.keras.layers.Layer):
    def __init__(self, units, activation=None, dropout_rate=0.0, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = activation
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dense = tf.keras.layers.Dense(units, use_bias=False)

    def call(self, X, A_norm, training=False):
        H = tf.matmul(A_norm, self.dense(X))
        if self.activation:
            H = self.activation(H)
        return self.dropout(H, training=training)

# -------------------- FIXED GCN MODEL --------------------
class HardGCN(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.gcn1 = GCNLayer(64, activation=tf.nn.relu, dropout_rate=0.3)
        self.gcn2 = GCNLayer(32, activation=tf.nn.relu, dropout_rate=0.3)
        self.output_layer = GCNLayer(2)

    def call(self, X, A_norm, training=False):
        H = self.gcn1(X, A_norm, training=training)
        H = self.gcn2(H, A_norm, training=training)
        return self.output_layer(H, A_norm, training=training)

# -------------------- LOAD MODEL WEIGHTS SAFELY --------------------
model = HardGCN()

# build model with dummy data so weights can be loaded
dummy_X = np.zeros((1, 12), dtype=np.float32)
dummy_A = np.array([[1.0]], dtype=np.float32)
model(dummy_X, dummy_A, training=False)

model.load_weights("hard_gcn_model.h5")   # ✅ loads correctly
print("✅ Model weights loaded successfully!")

# -------------------- PERMISSION VECTOR FEATURES --------------------
VECTOR_FEATURES = [
    "permission:android.permission.INTERNET",
    "permission:android.permission.ACCESS_NETWORK_STATE",
    "permission:android.permission.SEND_SMS",
    "permission:android.permission.RECEIVE_SMS",
    "permission:android.permission.READ_SMS",
    "permission:android.permission.READ_CONTACTS",
    "permission:android.permission.WRITE_EXTERNAL_STORAGE",
    "permission:android.permission.CAMERA",
    "permission:android.permission.ACCESS_FINE_LOCATION"
]

def extract_features(apk_path):
    a, d, dx = AnalyzeAPK(apk_path)
    perms = a.get_permissions()

    vector = np.zeros(len(VECTOR_FEATURES), dtype=float)
    for i, feat in enumerate(VECTOR_FEATURES):
        perm = feat.split(":")[1]
        if perm in perms:
            vector[i] = 1.0

    X = np.pad(vector, (0, 12 - len(vector))).reshape(1, 12).astype(np.float32)
    A_norm = np.array([[1.0]], dtype=np.float32)
    return X, A_norm

# -------------------- FLASK API --------------------
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    if "apk" not in request.files:
        return jsonify({"error": "Upload an APK using form key 'apk'"}), 400

    apk_file = request.files["apk"]
    apk_path = "/tmp/uploaded.apk"
    apk_file.save(apk_path)

    X, A_norm = extract_features(apk_path)
    logits = model(X, A_norm, training=False).numpy()
    probs = tf.nn.softmax(logits, axis=1).numpy()[0]

    malware_prob = float(probs[1])
    result = "MALWARE" if malware_prob >= 0.5 else "BENIGN"

    return jsonify({
        "result": result,
        "malware_probability": round(malware_prob, 4)
    })

@app.route("/")
def home():
    return "✅ HardGCN Malware Detection API Running"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
