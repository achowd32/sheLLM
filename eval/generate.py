import sys
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.models import load_model

sys.path.append("..")
from arch import architecture

# arguments
filename = sys.argv[1]
max_tok = int(sys.argv[2])
vocab_size = 128

# model
model = load_model(filename, custom_objects={"GPTLanguageModel": architecture.GPTLanguageModel})

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
