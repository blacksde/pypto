# Import the compiled extension
import sys
import os

# Try to load the compiled extension
try:
    from .lib import pypto_core as _core
except ImportError:
    # Fallback: try to load from build directory
    build_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'build', 'python', 'bindings')
    if build_path not in sys.path:
        sys.path.insert(0, build_path)
    import pypto_core as _core

# Re-export everything from the compiled module
from pypto_core import *
