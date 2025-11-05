# models/create_and_save_gcn.py

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model


OUT_PATH = os.path.join(os.path.dirname(__file__), "gnn_model.h5")
ROOT = os.path.join(os.path.dirname(__file__), "..")


def build_gnn(num_features):
    features = Input(shape=(num_features,), name="features")
    adj = Input(shape=(1, 1), name="adj")

    h = Dense(16, use_bias=False)(features)
    h = tf.nn.relu(h)
    logits = Dense(2, use_bias=False)(h)

    logits = Lambda(
        lambda args: tf.matmul(args[1], tf.expand_dims(args[0], -1))[:, :, 0]
    )([logits, adj])

    model = Model(inputs=[features, adj], outputs=logits)
    return model


def train_and_save():
    with open(os.path.join(ROOT, "model_features.json")) as f:
        features = json.load(f)

    num_features = len(features)
    model = build_gnn(num_features)

    X = np.random.randint(0, 2, (200, num_features))
    A = np.ones((200, 1, 1), dtype=np.float32)

    y = (X[:, 2] | X[:, 9]).astype(int)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.01),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )

    model.fit([X, A], y, epochs=20, batch_size=16)

    model.save(OUT_PATH)
    print("✅ Saved:", OUT_PATH)


if __name__ == "__main__":
    train_and_save()
