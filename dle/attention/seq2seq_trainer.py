import logging
import os
import time
from typing import Union, Optional

import gin
import tensorflow as tf
from tensorflow.keras.optimizers import Optimizer

from dle.attention import Decoder, Encoder

logger = logging.getLogger(__name__)


@gin.configurable
class Seq2SeqTrainer:

    def __init__(self,
                 encoder: Encoder,
                 decoder: Decoder,
                 optimizer: Optimizer = tf.keras.optimizers.Adam(),
                 checkpoint_prefix: Optional[Union[str, os.PathLike]] = None):
        self._encoder = encoder
        self._decoder = decoder
        self._optimizer = optimizer
        self._checkpoint_prefix = checkpoint_prefix
        if self._checkpoint_prefix is not None:
            self._checkpoint = tf.train.Checkpoint(encoder=self._encoder,
                                                   decoder=self._decoder,
                                                   optimizer=self._optimizer)

        self._loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                          reduction='none')
        self._mean_metric = tf.keras.metrics.Mean()

    def train(self,
              train_data: tf.data.Dataset,
              epochs: int):
        for e in tf.range(epochs):
            logger.info(f'Starting epoch {e}')
            start = time.time()
            for i, (input_sequence, target_sequence) in train_data.enumerate():
                loss = self._train_step(input_sequence,
                                        target_sequence)
                self._mean_metric.update_state(loss)
                logger.debug(f'Batch{i}: loss {loss}')

            logger.info(f'Mean loss: {self._mean_metric.result()}')
            logger.info(f'Epoch took: {time.time() - start}s')
            self._mean_metric.reset_states()
            if self._checkpoint_prefix is not None:
                logger.info('Model checkpoint')
                self._checkpoint.save(file_prefix=self._checkpoint_prefix)

    def load_trained(self):
        self._checkpoint.restore(tf.train.latest_checkpoint(self._checkpoint_prefix))

    @tf.function
    def _train_step(self,
                    input_seq: tf.Tensor,
                    target_seq: tf.Tensor
                    ) -> tf.Tensor:
        loss = 0.
        with tf.GradientTape() as tape:
            enc_out, dec_state = self._encoder(input_seq)
            for t in tf.range(tf.shape(target_seq)[1] - 1):
                dec_input = target_seq[:, t]  # Using teacher forcing
                logits, dec_state = self._decoder(dec_input, dec_state, enc_out)
                dec_output = target_seq[:, t + 1]
                loss = loss + self._calculate_loss(dec_output, logits)

            loss = loss / tf.cast(tf.shape(target_seq)[0], tf.float32)
            loss = loss / tf.cast(tf.shape(target_seq)[1], tf.float32)

        variables = self._encoder.trainable_variables + self._decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        self._optimizer.apply_gradients(zip(gradients, variables))
        return loss

    def _calculate_loss(self,
                        target: tf.Tensor,
                        logits: tf.Tensor):
        mask = tf.math.logical_not(tf.math.equal(target, 0))
        loss = self._loss_object(target, logits)
        loss = loss * tf.cast(mask, dtype=loss.dtype)
        loss = tf.reduce_mean(loss)
        return loss
