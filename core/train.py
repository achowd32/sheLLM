import sys
import json
import tensorflow as tf
import os
from tensorflow.keras.models import load_model

sys.path.append("..")
from arch import architecture

# initialize arguments
eval_interval = int(sys.argv[1])
max_iters = int(sys.argv[2])
filename = sys.argv[3]

i = 0
vocab_size = 128

# prepare model; load or initialize full Keras model in one file
# model.optimizer will be used for gradient updates
# ensure .keras extension for model file
if not filename.endswith('.keras'):
    filename = filename + '.keras'
model_path = os.path.join('..', filename)
# load or initialize model and compile for optimizer state
if os.path.exists(model_path):
    model = load_model(model_path, custom_objects={'GPTLanguageModel': architecture.GPTLanguageModel})
else:
    model = architecture.GPTLanguageModel(vocab_size)
    # build & compile to initialize weights and optimizer state
    _ = model(tf.zeros((1, 1), dtype=tf.int32))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    )

# main training loop: read batches from stdin
for line in sys.stdin:
    # parse JSON batch
    batch = json.loads(line)
    xb = tf.convert_to_tensor(batch["batch_x"], dtype=tf.int32)
    yb = tf.convert_to_tensor(batch["batch_y"], dtype=tf.int32)

    # periodic logging: save full model + optimizer in native Keras format
    if i % eval_interval == 0 or i == max_iters - 1:
        # logs saved as .keras model files
        log_path = f"../logs/{i}.keras"
        model.save(log_path, include_optimizer=True)
        print(i, flush=True)

    # (model+optimizer state already loaded)

    # forward and backward pass
    with tf.GradientTape() as tape:
        logits = model(xb, training=True)
        loss = model.compute_loss(logits, yb)
    grads = tape.gradient(loss, model.trainable_variables)
    # apply gradients via model.optimizer to keep optimizer state in sync
    model.optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # save updated full model+optimizer state in one HDF5 file
    model.save(model_path, include_optimizer=True)

    i += 1