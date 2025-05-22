import logging
import sys
import os
import threading

_RECORD_COUNTER = 0
_RECORD_COUNTER_LOCK = threading.Lock()

ROOT_LOGGER, FILE_HANDLER, STREAM_HANDLER = None, None, None
class CallOrderFilter(logging.Filter):
    """Filter that adds a sequential call_order attribute to each log record."""
    def filter(self, record):
        global _RECORD_COUNTER
        with _RECORD_COUNTER_LOCK:
            _RECORD_COUNTER += 1
            record.call_order = _RECORD_COUNTER
        return True

def setup_root_logger(log_file_name="workflow.log"):
    """Configure the one-and-only file handler, filter, etc., on the qrew root logger."""
    global ROOT_LOGGER, FILE_HANDLER
    print(f"Setting up logger")
    if ROOT_LOGGER is not None:
        return ROOT_LOGGER

    ROOT_LOGGER = logging.getLogger("qrew")
    ROOT_LOGGER.setLevel(logging.DEBUG)
    ROOT_LOGGER.propagate = False   # ⟵ stop it from bubbling up


    # 1) Filter that stamps call_order
    call_filter = CallOrderFilter()

    # 2) Single FileHandler — write once at startup, then always append
    base_dir = os.getcwd()
    log_file = os.path.join(base_dir, log_file_name)
    FILE_HANDLER = logging.FileHandler(log_file, mode="w")
    FILE_HANDLER.setLevel(logging.DEBUG)
    FILE_HANDLER.addFilter(call_filter)
    fmt = "[%(call_order)d] %(message)s - %(filename)s - %(funcName)s()"
    FILE_HANDLER.setFormatter(logging.Formatter(fmt))
    ROOT_LOGGER.addHandler(FILE_HANDLER)

    # 3) Optional console
    if os.getenv("LOG_TO_CONSOLE", "false").lower() == "true":
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.addFilter(call_filter)
        ch.setFormatter(logging.Formatter(fmt))
        ROOT_LOGGER.addHandler(ch)

    return ROOT_LOGGER

def get_logger(name=""):
    """
    Returns either:
      - the root 'qrew' logger, if name=="" or "qrew"
      - or a child 'qrew.<name>' logger
    """
    root = setup_root_logger()
    if name in ("", "qrew"):
        return root
    lg = logging.getLogger(f"qrew.{name}")
    lg.setLevel(logging.DEBUG)
    # send its records up to 'qrew' (where the handlers live)
    lg.propagate = True
    return lg

def reset_call_order():
    """Zero out the counter so the next log will be [1]."""
    global _RECORD_COUNTER
    with _RECORD_COUNTER_LOCK:
        _RECORD_COUNTER = 0