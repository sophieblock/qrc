import os, sys
import pytest
import logging
from qrew.util.log import get_logger,reset_call_order

SEP_STR = "#" * 150
qrew_logger = get_logger("qrew")  

# ——— YOUR sys.path hack (if you must) ———
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)



# ——— then the runslow plugin ———
def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--showlogs",
        action="store_true",
        default=False,
        help="show qrew DEBUG logs even if tests pass",
    )
    parser.addoption(
        "--log_to_console",
        action="store_true",
        default=False,
        help="will log debugs to console"
    )

def pytest_configure(config):
    show = config.getoption("--showlogs")

    log_to_console = config.getoption("log_to_console")
    
    if show:
        # Re-attach a console handler at DEBUG level
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)
        fmt = logging.Formatter('[%(call_order)d] %(message)s - %(filename)s - %(funcName)s()')
        ch.setFormatter(fmt)
        qrew_logger.addHandler(ch)
    if log_to_console:
        console_formatter = logging.Formatter('[%(call_order)d] %(message)s - %(filename)s - %(funcName)s()')
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)

        console_handler.setFormatter(console_formatter)
        qrew_logger.addHandler(console_handler)

    # Your existing slow‐test marker setup
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):

    if config.getoption("--runslow"):
        return
    skip = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)

@pytest.fixture(autouse=True)
def log_test_name(request):
    # start fresh numbering for this test
    reset_call_order()
    
    qrew_logger.debug(f"{SEP_STR}\nStarting test: {request.node.name}")
