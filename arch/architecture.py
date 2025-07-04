import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from arch.hyperparameters import *


class Head(layers.Layer):
    """One head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = layers.Dense(head_size, use_bias=False)
        self.query = layers.Dense(head_size, use_bias=False)
        self.value = layers.Dense(head_size, use_bias=False)
        self.dropout = layers.Dropout(DROPOUT)

    def build(self, input_shape):
        self.tril = tf.linalg.band_part(tf.ones((BLOCK_SIZE, BLOCK_SIZE)), -1, 0)

    def call(self, x, training=False):
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        wei = tf.matmul(q, k, transpose_b=True) * tf.math.rsqrt(dk)
        mask = self.tril[:T, :T]
        mask = tf.reshape(mask, (1, T, T))
        wei = tf.where(mask == 0, tf.fill(tf.shape(wei), float('-inf')), wei)
        wei = tf.nn.softmax(wei, axis=-1)
        wei = self.dropout(wei, training=training)
        v = self.value(x)
        out = tf.matmul(wei, v)
        return out


class MultiHeadAttention(layers.Layer):
    """Multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = [Head(head_size) for _ in range(num_heads)]
        self.proj = layers.Dense(N_EMBD)
        self.dropout = layers.Dropout(DROPOUT)

    def call(self, x, training=False):
        out = tf.concat([h(x, training=training) for h in self.heads], axis=-1)
        out = self.proj(out)
        out = self.dropout(out, training=training)
        return out


class FeedForward(layers.Layer):
    """A simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = keras.Sequential([
            layers.Dense(4 * n_embd, activation='relu'),
            layers.Dense(n_embd),
            layers.Dropout(DROPOUT),
        ])

    def call(self, x, training=False):
        return self.net(x, training=training)


class Block(layers.Layer):
    """Transformer block: communication followed by computation"""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = layers.LayerNormalization()
        self.ln2 = layers.LayerNormalization()

    def call(self, x, training=False):
        x = x + self.sa(self.ln1(x), training=training)
        x = x + self.ffwd(self.ln2(x), training=training)
        return x


class GPTLanguageModel(keras.Model):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = layers.Embedding(vocab_size, N_EMBD)
        self.position_embedding_table = layers.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = [Block(N_EMBD, N_HEAD) for _ in range(N_LAYER)]
        self.ln_f = layers.LayerNormalization()
        self.lm_head = layers.Dense(vocab_size)

    def call(self, idx, targets=None, training=False):
        B = tf.shape(idx)[0]
        T = tf.shape(idx)[1]
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_indices = tf.range(T)
        pos_emb = self.position_embedding_table(pos_indices) # (T,C)
        pos_emb = tf.expand_dims(pos_emb, 0)
        x = tok_emb + pos_emb # (B,T,C)
        for block in self.blocks:
            x = block(x, training=training)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B,T,vocab_size)

        loss = None
        if targets is not None:
            logits_flat = tf.reshape(logits, [-1, tf.shape(logits)[-1]])
            targets_flat = tf.reshape(targets, [-1])
            loss = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(targets_flat, logits_flat, from_logits=True)
            )
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond, training=False)
            logits = logits[:, -1, :] # (B, vocab_size)
            probs = tf.nn.softmax(logits, axis=-1)
            idx_next = tf.random.categorical(tf.math.log(probs), num_samples=1)
            idx = tf.concat([idx, idx_next], axis=1)
        return idx