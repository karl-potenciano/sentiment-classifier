import tensorflow as tf 
from tensorflow.keras import layers, Sequential



class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock,self).__init__()
        self.attn_layer = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential(
            [layers.Dense(ff_dim, activation='relu'), layers.Dense(embed_dim)]
        )
        self.l_norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.l_norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.attn_layer(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.l_norm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(out1, training=training)
        return self.l_norm2(out1 + ffn_output)

    def get_config(self):
        return {"attn_layer": self.attn_layer, "ffn": self.ffn, "l_norm1": self.l_norm1, "l_norm2": self.l_norm2, "dropout1": self.dropout1, "dropout2": self.dropout2}