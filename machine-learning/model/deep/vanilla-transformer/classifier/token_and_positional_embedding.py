import tensorflow as tf
from tensorflow.keras import layers

class TokenAndPositionalEmbedding(layers.Layer):
    def __init__(self, seq_len, vocab_size, embed_dim):
        super(TokenAndPositionalEmbedding, self).__init__()
        self.token_embedding = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.positional_embedding = layers.Embedding(input_dim=seq_len, output_dim=embed_dim)

    def call(self, input):
        len = tf.shape(input)[-1]
        positions = tf.range(start=0, limit=len, delta=1)
        positions = self.positional_embedding(positions)
        tokens = self.token_embedding(input)
        return tokens + positions

    def get_config(self):
        return {"token_embedding": self.token_embedding, "positional_embedding": self.positional_embedding}