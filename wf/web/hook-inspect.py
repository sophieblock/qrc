# hook-inspect.py
import inspect

_orig_getsource = inspect.getsource

def _safe_getsource(obj):
    try:
        return _orig_getsource(obj)
    except (OSError, IOError):
        return ""  # or return some default

inspect.getsource = _safe_getsource