
import sys
import os

# assume this file is test/conftest.py, so project_root = parent of test/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

