import logging

import click
import gin

from dle.attention.dataset import create_dataset

logger = logging.getLogger(__name__)


@click.command(name='attention')
@click.option('--config', '-c', type=click.Path(exists=True), required=True)
@click.option('--datapath', '-d', type=click.Path(exists=True), required=True)
def run_attention(config: str, datapath: str):
    logger.info('Running attention model...')
    gin.parse_config_file(config)
    create_dataset(datapath)
