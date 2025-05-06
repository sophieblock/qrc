# -*- coding: utf-8 -*-
"""Workflow components submodule.

TODO

**BOEING PROPRIETARY**

Authors:
    Joel Thompson (richard.j.thompson3@boeing.com)

"""

# Set up the logger.
import os
from .util.log import get_logger
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ['LOG_TO_CONSOLE'] = 'false'
# os.environ['LOG_TO_CONSOLE'] = 'true'
logger = get_logger(__name__)

# Set up meta-information.

__author__ = 'Joel Thompson (richard.j.thompson3@boeing.com)'
__organization__ = 'Boeing Research & Technology'

#__all__ = ["auditing", "domain", "process", "resources", "engine", "paths"]

# Set up tests.


def load_tests(_loader, tests, _pattern):
    # tests.addTests(doctest.DocTestSuite(wf.auditing))
    return tests

# Main program.


if __name__ == '__main__':
    import doctest
    import unittest
    unittest.main()
