import click
import tensorflow as tf

from dle.attention import run_attention
from dle.utils.logging import setup_logging


@click.group()
@click.option('--debug', is_flag=True)
def cli(debug: bool):
    setup_logging(debug)
    if debug:
        tf.config.experimental_run_functions_eagerly(True)


cli.add_command(run_attention)


if __name__ == '__main__':
    cli()
