from flax import nnx
import jax
import jax.numpy as jnp


class ConvClassifier(nnx.Module):
  def __init__(self, rngs: nnx.Rngs):
    self.conv_1 = nnx.Conv(
      in_features=1, out_features=64, kernel_size=3, padding="SAME", rngs=rngs
    )
    self.layer_norm_1 = nnx.LayerNorm(num_features=64, epsilon=0.001, rngs=rngs)

    self.conv_2 = nnx.Conv(
      in_features=64, out_features=64, kernel_size=3, padding="SAME", rngs=rngs
    )
    self.layer_norm_2 = nnx.LayerNorm(num_features=64, epsilon=0.001, rngs=rngs)

    self.conv_3 = nnx.Conv(
      in_features=64, out_features=64, kernel_size=3, padding="SAME", rngs=rngs
    )
    self.layer_norm_3 = nnx.LayerNorm(num_features=64, epsilon=0.001, rngs=rngs)

    self.dense_1 = nnx.Linear(in_features=64, out_features=2, rngs=rngs)

  def __call__(self, x: jax.Array):
    x = self.conv_1(x)
    x = self.layer_norm_1(x)
    x = jax.nn.relu(x)

    x = self.conv_2(x)
    x = self.layer_norm_2(x)
    x = jax.nn.relu(x)

    x = self.conv_3(x)
    x = self.layer_norm_3(x)
    x = jax.nn.relu(x)

    x = jnp.mean(x, axis=(1,))  # global average pooling
    x = self.dense_1(x)
    x = jax.nn.softmax(x)
    return x


class TransformerBlock(nnx.Module):
  def __init__(
    self,
    embed_dim: int,
    num_heads: int,
    ff_dim: int,
    rngs: nnx.Rngs,
    rate: float = 0.1,
  ):
    self.attention = nnx.MultiHeadAttention(
      num_heads=num_heads,
      in_features=embed_dim,
      qkv_features=embed_dim,
      decode=False,
      rngs=rngs,
    )

    self.dense_1 = nnx.Linear(in_features=embed_dim, out_features=ff_dim, rngs=rngs)
    self.dense_2 = nnx.Linear(in_features=ff_dim, out_features=ff_dim, rngs=rngs)

    self.layer_norm_1 = nnx.LayerNorm(num_features=embed_dim, epsilon=1e-6, rngs=rngs)
    self.layer_norm_2 = nnx.LayerNorm(num_features=ff_dim, epsilon=1e-6, rngs=rngs)

    self.dropout_1 = nnx.Dropout(rate, rngs=rngs)
    self.dropout_2 = nnx.Dropout(rate, rngs=rngs)

  def __call__(self, inputs: jax.Array):
    x = self.attention(inputs, inputs)
    x = self.dropout_1(x)
    x_norm_1 = self.layer_norm_1(inputs + x)
    x = self.dense_1(x_norm_1)
    x = jax.nn.relu(x)
    x = self.dense_2(x)
    x = self.dropout_2(x)
    x = self.layer_norm_2(x_norm_1 + x)
    return x


class TokenAndPositionEmbedding(nnx.Module):
  def __init__(self, max_length: int, vocab_size: int, embed_dim: int, rngs: nnx.Rngs):
    self.token_emb = nnx.Embed(num_embeddings=vocab_size, features=embed_dim, rngs=rngs)
    self.pos_emb = nnx.Embed(num_embeddings=max_length, features=embed_dim, rngs=rngs)

  def __call__(self, x: jax.Array):
    maxlen = jnp.shape(x)[-1]
    positions = jnp.arange(start=0, stop=maxlen, step=1)
    positions = self.pos_emb(positions)
    x = self.token_emb(x)
    return x + positions


embed_dim = 32  # Embedding size for each token
num_heads = 2  # Number of attention heads
ff_dim = 32  # Hidden layer size in the feed forward network inside transformer
# TODO: make these kwargs
vocab_size = 20000  # Only consider the top 20k words
maxlen = 200  # Only consider the first 200 words of each movie review


class Transformer(nnx.Module):
  def __init__(self, rngs: nnx.Rngs):
    self.embedding_layer = TokenAndPositionEmbedding(
      maxlen, vocab_size, embed_dim, rngs=rngs
    )
    self.transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim, rngs=rngs)
    self.dropout1 = nnx.Dropout(0.1, rngs=rngs)
    self.dense1 = nnx.Linear(in_features=embed_dim, out_features=20, rngs=rngs)
    self.dropout2 = nnx.Dropout(0.1, rngs=rngs)
    self.dense2 = nnx.Linear(in_features=20, out_features=2, rngs=rngs)

  def __call__(self, x: jax.Array):
    x = self.embedding_layer(x)
    x = self.transformer_block(x)
    x = jnp.mean(x, axis=(1,))  # global average pooling
    x = self.dropout1(x)
    x = self.dense1(x)
    x = jax.nn.relu(x)
    x = self.dropout2(x)
    x = self.dense2(x)
    x = jax.nn.softmax(x)
    return x
