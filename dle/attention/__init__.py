import logging

import click

logger = logging.getLogger(__name__)


@click.command(name='attention')
def run_attention():
    logger.info('Running attention model')