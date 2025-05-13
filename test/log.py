import logging
import sys
import os
ROOT_LOGGER, FILE_HANDLER, STREAM_HANDLER = None, None, None
def get_logger(name):
    """Create and configure a logger."""
    global ROOT_LOGGER, FILE_HANDLER, STREAM_HANDLER
    # log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    log_format = '%(message)s - %(filename)s - %(funcName)s()'
    logger = logging.getLogger(name)

    if not logger.handlers:  # Avoid adding duplicate handlers
        logger.setLevel(logging.DEBUG)
        
        # File logging
        if getattr(sys, 'frozen', False):
            # one-file or one-dir bundle
            logger.propagate = True  
        
            base_dir = os.path.dirname(sys.executable)
        else:
            logger.propagate = False  # Prevent propagation to the root logger
        
            base_dir = os.getcwd()

        log_file = os.path.join(base_dir, "qrew.log")
        try:
            file_handler = logging.FileHandler(log_file, "w")
            file_handler.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter(log_format)
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
            # print(f"File logging set up to {log_file}")
        except Exception as e:
            print(f"Failed to set up file logging: {e}", file=sys.stderr)

        # Console logging (based on LOG_TO_CONSOLE)
        log_to_console = os.getenv('LOG_TO_CONSOLE', 'false').lower() == 'true'
        if log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.DEBUG)
            console_formatter = logging.Formatter(log_format)
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

    return logger

