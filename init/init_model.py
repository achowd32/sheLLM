import sys
import tensorflow as tf

sys.path.append("..")
from arch import architecture

# initialize arguments
learning_rate = float(sys.argv[1])
filename = sys.argv[2]

vocab_size = 128
# create a TensorFlow model
model = architecture.GPTLanguageModel(vocab_size)

# print the number of parameters in the model
print(model.count_params()/1e6, 'M parameters')

# create a TensorFlow optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# save model and optimizer state
ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
ckpt.write(f"../{filename}")
