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

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")

def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
# print(f"project_root: {project_root}")

@pytest.fixture(autouse=True)
def capture_and_maybe_show_logs(request, caplog):
    """
    1) Always set caplog to capture DEBUG for qrew loggers.
    2) After the test runs, if --showlogs was passed AND the test
       actually passed, dump the captured records to the terminal.
    """
    show = request.config.getoption("--showlogs")
    # only capture DEBUG+ from your 'qrew' namespace:
    caplog.set_level(logging.DEBUG, logger='qrew')

    # Pre-test: send the test name into your file logger (still goes to qrew.log)
    qrew_logger.debug(f"{SEP_STR}\nStarting test: {request.node.name}")

    yield  # run the test

    # Post-test: if they asked for it, and it passed, print the logs:
    rep = getattr(request.node, "rep_call", None)
    if show and rep and rep.passed:
        # write to pytest’s own terminal reporter (bypasses capture)
        terminal = request.config.pluginmanager.get_plugin("terminalreporter")
        terminal.write_line(f"\n{SEP_STR}")
        terminal.write_line(f"Logs for {request.node.name}:")
        for record in caplog.records:
            # this mirrors your file format: message – filename – funcName()
            terminal.write_line(
                f"{record.getMessage()} - {record.filename} - {record.funcName}()"
            )
# @pytest.fixture(autouse=True)
# def log_test_name(request):
#     """
#     Runs before *every* test.  Logs a big separator + the test’s name.
#     """
#     logger.debug(f"{SEP_STR}\nStarting test: {request.node.name}")
