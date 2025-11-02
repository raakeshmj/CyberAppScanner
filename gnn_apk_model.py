import numpy as np
import tensorflow as tf
import numpy as np
import pandas as pd
from spektral.datasets import citation
import matplotlib.pyplot as plt

num_nodes = 10
num_features = 9
num_classes = 2
hidden_units = 16
dropout_rate = 0.2
epochs = 200


np.random.seed(42)
X_int = np.random.randint(0, 2, (num_nodes, num_features)).astype(np.float32)
X = tf.convert_to_tensor(X_int, dtype=tf.float32)
y_int = np.array([0,1,0,1,0,1,0,1,0,1], dtype=np.int64)
y = tf.convert_to_tensor(y_int)


A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
for i in range(num_nodes-1):
    A[i, i+1] = 1
    A[i+1, i] = 1
np.fill_diagonal(A, 1)


D_inv_sqrt = np.diag(1.0 / np.sqrt(A.sum(axis=1)))
A_norm = D_inv_sqrt @ A @ D_inv_sqrt
A_norm = tf.convert_to_tensor(A_norm, dtype=tf.float32)


train_indices = tf.constant([0,1,2,3,4,5], dtype=tf.int64)
test_indices  = tf.constant([6,7,8,9], dtype=tf.int64)


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


model = SimpleGCN(hidden_units, num_classes, dropout_rate)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

for epoch in range(epochs):
    with tf.GradientTape() as tape:
        logits = model(X, A_norm, training=True)
        logits_train = tf.gather(logits, train_indices)
        y_train = tf.gather(y, train_indices)
        loss = loss_fn(y_train, logits_train)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    if epoch % 50 == 0:
        train_preds = tf.argmax(logits_train, axis=1)
        acc = tf.reduce_mean(tf.cast(train_preds == y_train, tf.float32))
        print(f"Epoch {epoch:03d} | Loss: {loss.numpy():.4f} | Train Acc: {acc.numpy():.4f}")


logits_test = tf.gather(model(X, A_norm, training=False), test_indices)
y_test = tf.gather(y, test_indices)
test_preds = tf.argmax(logits_test, axis=1)
test_acc = tf.reduce_mean(tf.cast(test_preds == y_test, tf.float32))
print("Test Accuracy:", test_acc.numpy())


my_vec = np.random.randint(0, 2, size=num_features).astype(np.float32)
X_custom = tf.convert_to_tensor(my_vec.reshape(1, -1), dtype=tf.float32)
A_custom = tf.constant([[1.0]], dtype=tf.float32)  # self-loop

logits_custom = model(X_custom, A_custom, training=False)
pred_custom = tf.argmax(logits_custom, axis=1).numpy()[0]

print("\nRandom vector:", my_vec)
print("Prediction for random vector:")
print("Non-Malicious" if pred_custom == 0 else "Malicious")



my_vec = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0], dtype=np.float32)
X_custom = tf.convert_to_tensor(my_vec.reshape(1, -1), dtype=tf.float32)
A_custom = tf.constant([[1.0]], dtype=tf.float32)

logits_custom = model(X_custom, A_custom, training=False)
pred_custom = tf.argmax(logits_custom, axis=1).numpy()[0]
print("The following APK is :\n")
print("Non-Malicious" if pred_custom == 0 else "Malicious")





dataset = citation.Cora()
graph = dataset[0]


X = graph.x.copy()


prob = 0.3
mask = (np.random.rand(*X.shape) < prob) & (X == 0)
X[mask] = 1


df_features = pd.DataFrame(X)
df_features.index.name = 'Node'


df_features_T = df_features.T
df_features_T.index.name = 'Feature'
df_features_T.columns.name = 'Node'

print("Node features ")
print(df_features_T.iloc[:10, :9])


y = np.argmax(graph.y, axis=1)
df_labels = pd.DataFrame({'Node': np.arange(len(y)), 'Label': y})
print("\nNode labels (first 10 nodes):")
print(df_labels.head(10))


A = graph.a.todense()
df_adj = pd.DataFrame(A)
df_adj.index.name = 'Node'
print("\nAdjacency matrix (first 10 nodes):")
print(df_adj.iloc[:10, :10])



plt.figure(figsize=(6,4))
plt.hist(y, bins=np.arange(y.max()+2)-0.5, edgecolor='black')
plt.xlabel("Class Label")
plt.ylabel("Number of nodes")
plt.title("Distribution of node classes in Cora")
plt.show()
