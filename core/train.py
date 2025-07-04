import sys
import json
import tensorflow as tf

sys.path.append("..")
from arch import architecture

# initialize arguments
eval_interval = int(sys.argv[1])
max_iters = int(sys.argv[2])
filename = sys.argv[3]

i = 0
vocab_size = 128

# prepare model and optimizer
model = architecture.GPTLanguageModel(vocab_size)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

# main training loop: read batches from stdin
for line in sys.stdin:
    # parse JSON batch
    batch = json.loads(line)
    xb = tf.convert_to_tensor(batch["batch_x"], dtype=tf.int32)
    yb = tf.convert_to_tensor(batch["batch_y"], dtype=tf.int32)

    # logging
    if i % eval_interval == 0 or i == max_iters - 1:
        log_path = f"../logs/{i}.ckpt"
        model.save_weights(log_path)
        print(i, flush=True)

    # restore previous state
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    ckpt.restore(f"../{filename}").expect_partial()

    # forward and backward pass
    with tf.GradientTape() as tape:
        logits, loss = model(xb, yb, training=True)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # save updated model+optimizer state
    ckpt.write(f"../{filename}")

    i += 1