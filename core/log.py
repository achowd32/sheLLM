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
model = load_model(file_name, custom_objects={"GPTLanguageModel": architecture.GPTLanguageModel})

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
