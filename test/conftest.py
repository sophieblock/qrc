import os, sys
import pytest
import logging
from qrew import logger as qrew_logger   
SEP_STR = "#" * 150

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
        fmt = logging.Formatter('%(message)s - %(filename)s - %(funcName)s()')
        ch.setFormatter(fmt)
        qrew_logger.addHandler(ch)
    if log_to_console:
        log_format = '%(message)s - %(filename)s - %(funcName)s()'
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_formatter = logging.Formatter(log_format)
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

    qrew_logger.debug(f"{SEP_STR}\nStarting test: {request.node.name}")

# @pytest.fixture(autouse=True)
# def capture_and_maybe_show_logs(request, caplog):
#     """
#     1) Always set caplog to capture DEBUG for qrew loggers.
#     2) After the test runs, if --showlogs was passed AND the test
#        actually passed, dump the captured records to the terminal.
#     """
#     show = request.config.getoption("--showlogs")
#     # only capture DEBUG+ from your 'qrew' namespace:
#     caplog.set_level(logging.DEBUG, logger='qrew')

#     # Pre-test: send the test name into your file logger (still goes to qrew.log)
#     qrew_logger.debug(f"{SEP_STR}\nStarting test: {request.node.name}")

#     yield  # run the test

#     # Post-test: if they asked for it, and it passed, print the logs:
#     rep = getattr(request.node, "rep_call", None)
#     if show and rep and rep.passed:
#         # write to pytest’s own terminal reporter (bypasses capture)
#         terminal = request.config.pluginmanager.get_plugin("terminalreporter")
#         terminal.write_line(f"\n{SEP_STR}")
#         terminal.write_line(f"Logs for {request.node.name}:")
#         for record in caplog.records:
#             # this mirrors your file format: message – filename – funcName()
#             terminal.write_line(
#                 f"{record.getMessage()} - {record.filename} - {record.funcName}()"
#             )
