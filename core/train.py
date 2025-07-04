import tensorflow as tf
import json
import sys
import os

sys.path.append("..")
from arch import architecture

# initialize arguments
eval_interval = int(sys.argv[1])
max_iters = int(sys.argv[2])
filename = sys.argv[3]

# Setup
vocab_size = 128
base_path = os.path.join("..", filename)
model = architecture.GPTLanguageModel(vocab_size)
_ = model(tf.zeros((1, 1), dtype=tf.int32))
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

# Checkpoint logic
ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, base_path, max_to_keep=1)
status = ckpt.restore(ckpt_manager.latest_checkpoint)
status.expect_partial()

# Graph-mode training step
@tf.function
def train_step(xb, yb):
    with tf.GradientTape() as tape:
        logits = model(xb, training=True)
        loss = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(yb, logits, from_logits=True)
        )
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

# Main loop
i = 0
for line in sys.stdin:
    batch = json.loads(line)
    xb = tf.convert_to_tensor(batch["batch_x"], dtype=tf.int32)
    yb = tf.convert_to_tensor(batch["batch_y"], dtype=tf.int32)

    # Training step
    loss = train_step(xb, yb)

    # Save after every iteration to persist optimizer state
    ckpt_manager.save()

    # Optional periodic log checkpoint
    if i % eval_interval == 0 or i == max_iters - 1:
        log_path = f"../logs/{i}"
        log_ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
        log_manager = tf.train.CheckpointManager(log_ckpt, log_path, max_to_keep=1)
        log_manager.save()
        print(f"{i}", flush=True)

    i += 1