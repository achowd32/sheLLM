import sys
import json
import tensorflow as tf
import os
from tensorflow.keras.models import load_model

sys.path.append("..")
from arch import architecture

# load arguments
file_name = sys.argv[1]
eval_iters = int(sys.argv[2])
vocab_size = 128
# model
model = architecture.GPTLanguageModel(vocab_size)
_ = model(tf.zeros((1, 1), dtype=tf.int32))  # Build model
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

# restore checkpoint
ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, file_name, max_to_keep=1)
status = ckpt.restore(ckpt_manager.latest_checkpoint)
status.expect_partial()

loss_sum = 0.0
for line in sys.stdin:  # read in one batch at a time
    # load from json and convert to tensor
    data = json.loads(line)
    X = tf.convert_to_tensor(data["batch_x"], dtype=tf.int32)
    Y = tf.convert_to_tensor(data["batch_y"], dtype=tf.int32)

    # calculate loss and add to loss_sum
    logits = model(X, training=False)
    loss = model.compute_loss(logits, Y)
    loss_sum += loss.numpy()

# print average loss
print(loss_sum / eval_iters)
