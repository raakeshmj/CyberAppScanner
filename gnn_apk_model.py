import numpy as np
import tensorflow as tf


num_apks = 200
num_features = 13
num_classes = 2
hidden_units = 32
dropout_rate = 0.2
epochs = 100

np.random.seed(42)
X_int = np.random.randint(0, 2, (num_apks, num_features))


degree_feat = X_int.sum(axis=1, keepdims=True)
X_int = np.hstack([X_int, degree_feat])
X = tf.constant(X_int.astype(np.float32))


y = np.zeros(num_apks, dtype=np.int32)
y[(X_int[:, 2] == 1) | (X_int[:, 3] == 1)] = 1
y = tf.constant(y, dtype=tf.int32)

A = np.zeros((num_apks, num_apks), dtype=np.float32)
for i in range(num_apks):
    for j in range(i, num_apks):
        if np.any(X_int[i] & X_int[j]):  # share any permission/API
            A[i, j] = 1
            A[j, i] = 1
np.fill_diagonal(A, 1)

D_inv_sqrt = np.diag(1.0 / np.sqrt(A.sum(axis=1)))
A_norm = D_inv_sqrt @ A @ D_inv_sqrt
A_norm = tf.constant(A_norm, dtype=tf.float32)

train_mask = np.zeros(num_apks, dtype=bool)
train_mask[:int(0.8*num_apks)] = True
np.random.shuffle(train_mask)
train_indices = np.where(train_mask)[0]
test_indices = np.where(~train_mask)[0]


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

model = SimpleGCN(hidden_units=hidden_units, num_classes=num_classes, dropout_rate=dropout_rate)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

for epoch in range(epochs):
    with tf.GradientTape() as tape:
        logits = model(X, A_norm, training=True)

       
        train_indices_tf = tf.constant(train_indices, dtype=tf.int32)
        test_indices_tf = tf.constant(test_indices, dtype=tf.int32)

        
        y_train_gathered = tf.gather(y, train_indices_tf)
        logits_train_gathered = tf.gather(logits, train_indices_tf)

        
        loss = loss_fn(y_train_gathered, logits_train_gathered)

    
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    
    train_preds = tf.argmax(logits_train_gathered, axis=1, output_type=tf.int32)
    train_acc = tf.reduce_mean(tf.cast(train_preds == y_train_gathered, tf.float32))

    logits_test_gathered = tf.gather(logits, test_indices_tf)
    y_test_gathered = tf.gather(y, test_indices_tf)
    test_preds = tf.argmax(logits_test_gathered, axis=1, output_type=tf.int32)
    test_acc = tf.reduce_mean(tf.cast(test_preds == y_test_gathered, tf.float32))

    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d} | Loss: {loss.numpy():.4f} | Train Acc: {train_acc.numpy():.4f} | Test Acc: {test_acc.numpy():.4f}")



final_preds = tf.argmax(model(X, A_norm, training=False), axis=1, output_type=tf.int32)
print("Final predictions:", final_preds.numpy())