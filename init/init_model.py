import sys
import tensorflow as tf
import os
from tensorflow.keras.models import load_model  # optionally for future loading

sys.path.append("..")
from arch import architecture

# initialize arguments
learning_rate = float(sys.argv[1])
filename = sys.argv[2]
# ensure native Keras format extension
if not filename.endswith('.keras'):
    filename = filename + '.keras'
model_path = os.path.join('..', filename)

vocab_size = 128
# create a TensorFlow model
model = architecture.GPTLanguageModel(vocab_size)
# build model (dummy forward pass) to initialize weights
_ = model(tf.zeros((1, 1), dtype=tf.int32))
# print the number of parameters in the model
print(model.count_params()/1e6, 'M parameters')

# create a TensorFlow optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# compile and save initial model + optimizer in one file
model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
)

# perform one manual training step to initialize optimizer slots
# dummy data for a single batch
x_dummy = tf.zeros((1, 1), dtype=tf.int32)
y_dummy = tf.zeros((1, 1), dtype=tf.int32)
with tf.GradientTape() as tape:
    logits = model(x_dummy, training=True)
    loss = model.compute_loss(logits, y_dummy)
grads = tape.gradient(loss, model.trainable_variables)
model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
model.save(model_path, include_optimizer=True)
