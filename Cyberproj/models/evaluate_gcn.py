import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, accuracy_score, roc_auc_score,
    roc_curve
)

# --------------------
# ADVANCED GCN LAYER
# --------------------
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


# --------------------
# ADVANCED GCN MODEL
# --------------------
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


# --------------------
# GENERATE HARD DATASET
# --------------------
def generate_hard_dataset(N=400, F=12):
    X = np.random.randint(0, 2, (N, F)).astype(np.float32)

    # Graph: connect samples by Hamming similarity (harder)
    A = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(i + 1, N):
            d = np.sum(X[i] != X[j])
            if d <= 3:
                A[i, j] = A[j, i] = 1

    np.fill_diagonal(A, 1)

    # Normalize adjacency
    D_inv = np.diag(1.0 / np.sqrt(A.sum(axis=1)))
    A_norm = D_inv @ A @ D_inv

    # Harder label rule (nonlinear): malware if
    # (feature3 AND feature5) OR (feature2 XOR feature7)
    y = (
        ((X[:, 3] * X[:, 5]) > 0)
        | ((X[:, 2].astype(int) ^ X[:, 7].astype(int)) > 0)
    ).astype(int)

    return X, A_norm.astype(np.float32), y.astype(np.int64)


# --------------------
# TRAINING LOOP
# --------------------
def train_hard_model(X, A_norm, y, epochs=250):
    model = HardGCN()

    optimizer = tf.keras.optimizers.Adam(0.005)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Build model
    model(X, A_norm, training=False)

    for epoch in range(1, epochs + 1):
        with tf.GradientTape() as tape:
            logits = model(X, A_norm, training=True)
            loss = loss_fn(y, logits)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if epoch % 25 == 0:
            preds_train = tf.argmax(logits, axis=1).numpy()
            acc = np.mean(preds_train == y)
            print(f"[Epoch {epoch}] Loss={loss.numpy():.4f} | Acc={acc:.4f}")

    return model


# --------------------
# EVALUATION
# --------------------
def evaluate_hard(model, X, A_norm, y, threshold=0.5):
    logits = model(X, A_norm, training=False).numpy()
    probs = tf.nn.softmax(logits, axis=1).numpy()[:, 1]

    y_pred = (probs >= threshold).astype(int)

    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    acc = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, probs)

    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn + 1e-9)

    print("\n========== GCN EVALUATION ==========")
    print(f"Accuracy:            {acc:.4f}")
    print(f"Precision:           {precision:.4f}")
    print(f"Recall:              {recall:.4f}")
    print(f"F1 Score:            {f1:.4f}")
    print(f"AUC:                 {auc:.4f}")
    print(f"False Positive Rate: {fpr:.4f}")
    print("Confusion Matrix:\n", cm)
    print("==========================================\n")

    return fpr, auc


# --------------------
# MAIN FLOW
# --------------------
if __name__ == "__main__":
    X, A_norm, y = generate_hard_dataset()
    print("Dataset Ready")

    model = train_hard_model(X, A_norm, y)
    print("Training Complete")

    evaluate_hard(model, X, A_norm, y)
