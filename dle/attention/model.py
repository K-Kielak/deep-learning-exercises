from typing import Tuple, Any

import gin
import tensorflow as tf


@gin.configurable
class Encoder(tf.keras.Model):

    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int = 256,
                 units: int = 1024):
        super(Encoder, self).__init__()
        self._embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                                    name='EmbeddingLayer')
        # TODO bidirectional?
        self._recurrent_cell = tf.keras.layers.GRU(units,
                                                   return_sequences=True,
                                                   return_state=True,
                                                   recurrent_initializer='glorot_uniform')

    def call(self,
             input_sequence: tf.Tensor,
             **kwargs: Any
             ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Processes input sentence and encodes it.

        Args:
            input_sequence: A tensor containing input sequence of shape
                [batch_size, sequence_length].

        Returns:
            A pair (tuple) of tensors:
                1. Encoded sequence of shape
                    [batch_size, sequence_length, hidden_dim].
                2. Hidden recurrent state of the encoder of shape:
                    [batch_size, hidden_dim].
        """
        inp = self._embedding(input_sequence)
        out, state = self._recurrent_cell(inp)
        return out, state


@gin.configurable
class Decoder(tf.keras.Model):

    def __init__(self,
                 vocab_size: int,
                 attention: tf.keras.layers.Layer,
                 embedding_dim: int = 256,
                 units: int = 1024):
        super(Decoder, self).__init__()

        self._embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self._recurrent_cell = tf.keras.layers.GRUCell(units,
                                                      recurrent_initializer='glorot_uniform')
        self._attention = attention
        self._output_layer = tf.keras.layers.Dense(vocab_size)

    def __call__(self,
                 last_dec_word: tf.Tensor,
                 prev_state: tf.Tensor,
                 encoder_out: tf.Tensor,
                 **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
        """Produces decoded output (translation).

        Args:
            last_dec_word: Last decoded word of shape [batch_size].
            prev_state: Previous hidden recurrent state of shape [batch_size, hidden_dim].
            encoder_out: Encoded input sequence of shape
                [batch_size, enc_seq_length, hidden_dim].

        Returns:
            A pair (tuple) of tensors:
                1. Multinomial logits for the next decoded word in the
                    sequence of shape [batch_size, vocab_size].
                2. Hidden recurrent state of the decoder of shape:
                    [batch_size, hidden_dim]
        """
        last_dec_word = self._embedding(last_dec_word)
        state, _ = self._recurrent_cell(last_dec_word, [prev_state])
        # state has shape [batch_size, hidden_dim], expand it to
        # [batch_size, 1, hidden_dim] to fit attention requirements
        context = self._attention(tf.expand_dims(state, 1), encoder_out)
        # context has shape [batch_size, 1, hidden_dim],
        # slice to get rid of redundant 1 in the middle
        output = tf.concat((state, context[:, 0, :]), axis=-1)
        output = self._output_layer(output)
        return output, state


class LuongAttention(tf.keras.layers.Layer):
    """Basic dot-product attention"""

    def __call__(self,
                 queries: tf.Tensor,
                 values: tf.Tensor,
                 **kwargs: Any
                 ) -> tf.Tensor:
        """Applies LuongAttention on tensors of queries and values.

        Args:
            queries: Queries tensor of shape [batch_size, queries_num, hidden_dim].
            values: Values/keys tensor of shape [batch_size, values_num, hidden_dim].

        Returns:
            Attention context tensor of shape [batch_size, queries_num, encoded_dim].
        """
        scores = tf.matmul(queries, tf.transpose(values))
        attention_weights = tf.nn.softmax(scores)
        return tf.matmul(attention_weights, values)


@gin.configurable
class BahdanauAttention(tf.keras.layers.Layer):
    """General additive attention"""

    def __init__(self,
                 units: int = 1024):
        super(BahdanauAttention, self).__init__()
        self._w1 = tf.keras.layers.Dense(units, use_bias=False)
        self._w2 = tf.keras.layers.Dense(units, use_bias=False)
        self._v = tf.keras.layers.Dense(1, use_bias=False)

    def __call__(self,
                 queries: tf.Tensor,
                 values: tf.Tensor,
                 **kwargs: Any
                 ) -> tf.Tensor:
        """Applies BahdanauAttention on tensors of queries and values.

        Args:
            queries: Queries tensor of shape [batch_size, queries_num, hidden_dim].
            values: Values/keys tensor of shape [batch_size, values_num, hidden_dim].

        Returns:
            Attention context tensor of shape [batch_size, queries_num, hidden_dim].
        """
        # Expand dims of queries and values to get
        # exp_queries: [batch_size, queries_num, 1, hidden_dim]
        # exp_values: [batch_size, 1, values_num, hidden_dim]
        exp_queries = tf.expand_dims(queries, 2)
        exp_values = tf.expand_dims(values, 1)

        added_qv = self._w1(exp_queries) + self._w2(exp_values)
        scores = self._v(tf.tanh(added_qv))
        # Scores shape is [batch_size, queries_num, values_num, 1]
        # Perform slice to get rid of redundant 1 at the end
        scores = scores[:, :, :, 0]
        attention_weights = tf.nn.softmax(scores)
        return tf.matmul(attention_weights, values)





