import logging


def setup_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    message_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    log_formatter = logging.Formatter(fmt=message_format)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(level)
    logger = logging.getLogger('dle')
    logger.setLevel(level)
    logger.addHandler(console_handler)
    logger.info(f'Debug mode is {"on" if debug else "off"}')
