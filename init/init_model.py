import sys
import tensorflow as tf
import os

sys.path.append("..")
from arch import architecture

# args
learning_rate = float(sys.argv[1])
filename = sys.argv[2]

# paths
base_path = os.path.join("..", filename)

# model and optimizer
vocab_size = 128
model = architecture.GPTLanguageModel(vocab_size)
_ = model(tf.zeros((1, 1), dtype=tf.int32))  # Build model
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# checkpoint
ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, base_path, max_to_keep=1)

# dummy step to initialize optimizer slots
x_dummy = tf.zeros((1, 1), dtype=tf.int32)
y_dummy = tf.zeros((1, 1), dtype=tf.int32)
with tf.GradientTape() as tape:
    logits = model(x_dummy, training=True)
    loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_dummy, logits, from_logits=True))
    
grads = tape.gradient(loss, model.trainable_variables)
optimizer.apply_gradients(zip(grads, model.trainable_variables))

# save checkpoint
ckpt_manager.save()