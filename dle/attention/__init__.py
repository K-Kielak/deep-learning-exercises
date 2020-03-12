import logging

import click
import gin
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer

from dle.attention.dataset import create_dataset
from dle.attention.model import Encoder, BahdanauAttention, Decoder
from dle.attention.seq2seq_trainer import Seq2SeqTrainer

logger = logging.getLogger(__name__)


@click.command(name='attention')
@click.option('--config', '-c', type=click.Path(exists=True), required=True)
@click.option('--datapath', '-d', type=click.Path(exists=True), required=True)
@click.option('--epochs', '-e', type=int, default=10)
def run_attention(config: str, datapath: str, epochs: int):
    logger.info('Running attention model...')
    gin.parse_config_file(config)
    train_dataset, val_dataset, inp_tokenizer, targ_tokenizer = create_dataset(datapath)
    encoder, decoder = create_model(len(inp_tokenizer.word_index) + 1,
                                    len(targ_tokenizer.word_index) + 1)
    trainer = Seq2SeqTrainer(encoder, decoder)
    trainer.train(train_dataset,
                  targ_tokenizer.word_index['<start>'],
                  epochs)
    trainer.train(train_dataset, epochs)
    evaluate(val_dataset, inp_tokenizer, targ_tokenizer, encoder, decoder)


def create_model(inp_vocab_size, targ_vocab_size):
    encoder = Encoder(inp_vocab_size)
    attention = BahdanauAttention()
    decoder = Decoder(targ_vocab_size, attention)
    return encoder, decoder


def evaluate(dataset: tf.data.Dataset,
             inp_tokenizer: Tokenizer,
             targ_tokenizer: Tokenizer,
             encoder: Encoder,
             decoder: Decoder):

    logger.info('Evaluation start...')
    for i, (inp, targ) in dataset.enumerate():
        out = evaluate_sequence(inp,
                                targ,
                                targ_tokenizer.word_index['<end>'],
                                encoder,
                                decoder)
        print(f'Sequence{i}\n'
              f'Input sentence: {tensor_to_word(inp, inp_tokenizer)}\n'
              f'Expected output: {tensor_to_word(targ, targ_tokenizer)}\n'
              f'Actual output: {tensor_to_word(out, targ_tokenizer)}\n'
              f'----------------------------------------------------')


def evaluate_sequence(input_seq: tf.Tensor,
                      target_seq: tf.Tensor,
                      end_token: int,
                      encoder: Encoder,
                      decoder: Decoder):
    result = tf.TensorArray(tf.int32, size=tf.shape(target_seq)[0])
    dec_input = tf.expand_dims(target_seq[0], 0)

    enc_out, dec_state = encoder(tf.expand_dims(input_seq, 0))
    for t in tf.range(tf.shape(target_seq)[0]):
        logits, dec_state = decoder(dec_input, dec_state, enc_out)
        # Recursively feed model output back to the model
        dec_input = tf.argmax(logits, axis=1, output_type=tf.int32)
        result.write(t, dec_input)
        if dec_input == end_token:
            return result.gather(tf.range(t))[:, 0]

    return result.concat()


def tensor_to_word(tensor: tf.Tensor, tokenizer: Tokenizer) -> str:
    return ' '.join([tokenizer.index_word[t] for t in tensor.numpy() if t != 0])
