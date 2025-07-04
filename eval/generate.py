import sys
import tensorflow as tf
import numpy as np
import os

sys.path.append("..")
from arch import architecture

# arguments
filename = sys.argv[1]
max_tok = int(sys.argv[2])
vocab_size = 128

# model
model = architecture.GPTLanguageModel(vocab_size)
_ = model(tf.zeros((1, 1), dtype=tf.int32))  # Build model
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

# restore checkpoint
ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, os.path.join("..", filename), max_to_keep=1)
status = ckpt.restore(ckpt_manager.latest_checkpoint)
status.expect_partial()

# read prompt
prompt = sys.stdin.read().strip()
if prompt:
    context = np.array([[int(n) for n in prompt.split()]], dtype=np.int32)
else:
    context = np.zeros((1, 1), dtype=np.int32)

# generate tokens
tokens = model.generate(tf.convert_to_tensor(context), max_tok)[0].numpy().tolist()

# print tokens
print(" ".join(str(t) for t in tokens))