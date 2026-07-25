import logging
import sys
import os

def setup_logger(name: str = "dermaid") -> logging.Logger:
    """
    Create and configure a logger.

    Level can be controlled via environment variable `DERMAID_LOG_LEVEL`
    (DEBUG, INFO, WARNING, ERROR, CRITICAL). Default: INFO.

    If `DERMAID_LOG_FILE` is set, logs are also written to that file.
    """
    logger = logging.getLogger(name)
    if logger.handlers:  # already configured
        return logger

    logger.setLevel(os.environ.get("DERMAID_LOG_LEVEL", "INFO").upper())

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)  # Let logger level filter
    console_format = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # Optional file handler
    log_file = os.environ.get("DERMAID_LOG_FILE")
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            "[%(asctime)s] %(levelname)s %(name)s (%(filename)s:%(lineno)d): %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger

# Module-level logger instance
logger = setup_logger()