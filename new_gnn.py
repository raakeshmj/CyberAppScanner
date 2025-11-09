import argparse
import numpy as np
from androguard.misc import AnalyzeAPK
import tensorflow as tf

# ---- IMPORT CUSTOM GCN CODE ----
class GCNLayer(tf.keras.layers.Layer):
    def __init__(self, units, activation=None, dropout_rate=0.0):
        super().__init__()
        self.units = units
        self.activation = activation
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dense = tf.keras.layers.Dense(units, use_bias=False)

    def call(self, X, A_norm, training=False):
        H = tf.matmul(A_norm, self.dense(X))
        if self.activation:
            H = self.activation(H)
        H = self.dropout(H, training=training)
        return H

class HardGCN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.gcn1 = GCNLayer(64, activation=tf.nn.relu, dropout_rate=0.3)
        self.gcn2 = GCNLayer(32, activation=tf.nn.relu, dropout_rate=0.3)
        self.output_layer = GCNLayer(2)

    def call(self, X, A_norm, training=False):
        H = self.gcn1(X, A_norm, training=training)
        H = self.gcn2(H, A_norm, training=training)
        return self.output_layer(H, A_norm, training=training)


# ---- YOUR 9 PERMISSION FEATURES ----
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


def generate_permission_vector(apk_path):
    print(f"[*] Analyzing '{apk_path}'...")
    try:
        a, d, dx = AnalyzeAPK(apk_path)
        if not a:
            print("[!] Failed to parse APK.")
            return None

        requested = a.get_permissions()
        print(f" -> Permissions found: {len(requested)}")

        vector = np.zeros(len(VECTOR_FEATURES), dtype=float)
        for i, feat in enumerate(VECTOR_FEATURES):
            perm = feat.split(":")[1]
            if perm in requested:
                vector[i] = 1.0

        return vector

    except Exception as e:
        print(f"[!] Error analyzing APK: {e}")
        return None


def predict_malware(vector):
    # ---- LOAD MODEL ----
    model = tf.keras.models.load_model("hard_gcn_model.h5", custom_objects={"GCNLayer": GCNLayer, "HardGCN": HardGCN})

    # ---- PAD FEATURE VECTOR FROM 9 → 12 ----
    X = np.pad(vector, (0, 12 - len(vector)))  # pad zeros
    X = X.reshape(1, 12).astype(np.float32)

    # ---- CREATE 1×1 GRAPH ----
    A_norm = np.array([[1.0]], dtype=np.float32)

    # ---- RUN INFERENCE ----
    logits = model(X, A_norm, training=False).numpy()
    probs = tf.nn.softmax(logits, axis=1).numpy()[0]
    malware_prob = probs[1]

    print("\n========== Prediction ==========")
    print(f"Malware Probability: {malware_prob:.4f}")

    if malware_prob >= 0.5:
        print("🔴 **Result: MALWARE DETECTED**")
    else:
        print("🟢 **Result: BENIGN**")

    print("================================\n")


def main():
    parser = argparse.ArgumentParser(description="APK Malware Detection using HardGCN")
    parser.add_argument("apk_file", help="Path to target .apk")
    args = parser.parse_args()

    vector = generate_permission_vector(args.apk_file)
    if vector is not None:
        predict_malware(vector)


if __name__ == "__main__":
    import sys

    # If in Jupyter/Colab → avoid argparse conflict
    if "ipykernel" in sys.modules:
        class Args:
            apk_file = "sample.apk"   # <- put your apk path here
        args = Args()
        vector = generate_permission_vector(args.apk_file)
        if vector is not None:
            predict_malware(vector)
    else:
        main()
