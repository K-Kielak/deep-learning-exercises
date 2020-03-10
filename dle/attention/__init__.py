import logging

import click
import gin

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


def create_model(inp_vocab_size, targ_vocab_size):
    encoder = Encoder(inp_vocab_size)
    attention = BahdanauAttention()
    decoder = Decoder(targ_vocab_size, attention)
    return encoder, decoder
