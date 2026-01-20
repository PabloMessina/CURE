import logging
import sys
import os
from logging.handlers import RotatingFileHandler
import colorlog

# Define the color log format with alignment
# %-8s ensures the level name (INFO, DEBUG) takes up 8 spaces for alignment
COLOR_LOG_FORMAT = (
    "%(log_color)s%(asctime)s%(reset)s | "
    "%(log_color)s%(levelname)-8s%(reset)s | "
    "%(log_color)s%(name)s%(reset)s: "
    "%(message_log_color)s%(message)s"
)

# Standard format for files (clean, aligned, no color codes)
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s: %(message)s"

ANSI_MAGENTA_BOLD = "\033[1;35m"
ANSI_CYAN_BOLD = "\033[1;36m"
ANSI_YELLOW_BOLD = "\033[1;33m"
ANSI_RED_BOLD = "\033[1;31m"
ANSI_WHITE_BOLD = "\033[1;37m"
ANSI_BLACK_BOLD = "\033[1;30m"
ANSI_RESET = "\033[0m"

LOG_LEVEL = logging.INFO

def setup_logging(
    log_level=LOG_LEVEL,
    log_file=None,
    log_format=LOG_FORMAT,
    color_log_format=COLOR_LOG_FORMAT,
    use_console=True,
    use_color=True,
    date_format="%Y-%m-%d %H:%M:%S" # Added cleaner date format
):
    """
    Configures the root logger with aligned, colored output.
    """
    logger = logging.getLogger()
    logger.setLevel(log_level)

    if logger.hasHandlers():
        logger.handlers.clear()

    # Standard formatter for files
    file_formatter = logging.Formatter(log_format, datefmt=date_format)

    handlers = []

    # Console Handler
    if use_console:
        if use_color:
            console_formatter = colorlog.ColoredFormatter(
                color_log_format,
                datefmt=date_format,
                log_colors={
                    'DEBUG':    'cyan',
                    'INFO':     'green',
                    'WARNING':  'yellow',
                    'ERROR':    'red',
                    'CRITICAL': 'red,bg_white',
                },
                secondary_log_colors={
                    'message': {
                        'ERROR':    'red',
                        'CRITICAL': 'red'
                    }
                },
                style='%'
            )
        else:
            console_formatter = logging.Formatter(log_format, datefmt=date_format)

        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(console_formatter)
        handlers.append(console_handler)

    # File Handler
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        file_handler = RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5
        )
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)

    if not handlers:
        logger.addHandler(logging.NullHandler())
    else:
        for handler in handlers:
            logger.addHandler(handler)
            
    # Silence third-party noise explicitly in setup
    logging.getLogger("PyRuSH").setLevel(logging.ERROR)
    
    # Message for the user
    if handlers:
        # We use the root logger to confirm setup
        logging.info(f"Logging configured (Level: {logging.getLevelName(log_level)})")