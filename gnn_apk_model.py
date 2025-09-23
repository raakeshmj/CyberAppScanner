import numpy as np
import tensorflow as tf
from spektral.datasets import citation


dataset = citation.Cora()
graph = dataset[0]


X = tf.convert_to_tensor(graph.x, dtype=tf.float32)  #
y_onehot = tf.convert_to_tensor(graph.y, dtype=tf.float32)  
A = tf.convert_to_tensor(graph.a.todense(), dtype=tf.float32)  

y = tf.argmax(y_onehot, axis=1, output_type=tf.int64)  

num_nodes, num_features = X.shape
num_classes = y.numpy().max() + 1  # 7


A = A.numpy()
D_inv_sqrt = np.diag(1.0 / np.sqrt(A.sum(axis=1)))
A_norm = D_inv_sqrt @ A @ D_inv_sqrt
A_norm = tf.convert_to_tensor(A_norm, dtype=tf.float32)


train_mask = dataset.mask_tr
test_mask = dataset.mask_te
train_indices = tf.constant(np.where(train_mask)[0], dtype=tf.int64)
test_indices = tf.constant(np.where(test_mask)[0], dtype=tf.int64)


class SimpleGCN(tf.keras.Model):
    def __init__(self, hidden_units, num_classes, dropout_rate):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(hidden_units, use_bias=False)
        self.dense2 = tf.keras.layers.Dense(num_classes, use_bias=False)
        self.dropout_rate = dropout_rate

    def call(self, X, A, training=False):
        H = tf.matmul(A, self.dense1(X))
        H = tf.nn.relu(H)
        if training:
            H = tf.nn.dropout(H, rate=self.dropout_rate)
        Z = tf.matmul(A, self.dense2(H))
        return Z


model = SimpleGCN(hidden_units=32, num_classes=num_classes, dropout_rate=0.2)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


epochs = 200
for epoch in range(epochs):
    with tf.GradientTape() as tape:
        logits = model(X, A_norm, training=True)
        logits_train = tf.gather(logits, train_indices)
        y_train = tf.gather(y, train_indices)
        loss = loss_fn(y_train, logits_train)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    if epoch % 20 == 0:
        train_preds = tf.argmax(logits_train, axis=1)
        acc = tf.reduce_mean(tf.cast(train_preds == y_train, tf.float32))
        print(f"Epoch {epoch:03d} | Loss: {loss.numpy():.4f} | Train Acc: {acc.numpy():.4f}")

logits_test = tf.gather(model(X, A_norm, training=False), test_indices)
y_test = tf.gather(y, test_indices)
test_preds = tf.argmax(logits_test, axis=1)
test_acc = tf.reduce_mean(tf.cast(test_preds == y_test, tf.float32))
print("Test Accuracy:", test_acc.numpy())


my_vec = np.zeros(num_features, dtype=np.float32)  
X_custom = tf.convert_to_tensor(my_vec.reshape(1,-1), dtype=tf.float32)
A_custom = tf.constant([[1.0]], dtype=tf.float32)  

logits = model(X_custom, A_custom, training=False)
pred = tf.argmax(logits, axis=1).numpy()
print("Prediction for custom vector:", pred)
