"""Shortcuts for logging throughout the codebase."""

import logging

LOG_FORMAT = '[%(asctime)s]\t[%(levelname)s]  \t%(message)s'


def init_logger(args):
    """Initializes the logger object for the run session.

    Args:
        args (argparse.Namespace): Output from ArgumentParser.parse_args (more
            info in main.py)."""
    if args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(format=LOG_FORMAT,
                        level=level,
                        datefmt='%H:%M:%S')
    log('Logger initialized.')


def debug(text):
    """Logs a debugging message to the console."""
    logging.debug(text)


def log(text):
    """Logs an information message to the console."""
    logging.info(text)


def warn(text):
    """Logs a warning message to the console."""
    logging.warning(text)


def error(text):
    """Logs an error message to the console."""
    logging.error(text)
