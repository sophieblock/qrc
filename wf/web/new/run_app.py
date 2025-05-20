import logging
import sys
import traceback
from shiny import run_app

logging.basicConfig(level=logging.DEBUG)

try:
    # Your main application code here
    run_app(host='127.0.0.1', port=0, app_dir=".", launch_browser=True)
except Exception as e:
    logging.error("An error occurred: %s", e)
    logging.error("Traceback: %s", traceback.format_exc())
    sys.exit(1)
