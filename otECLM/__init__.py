"""
    otECLM --- An OpenTURNS module
    ==================================

    Contents
    --------
      'otECLM' is a module for OpenTURNS

"""

import sys
if sys.platform.startswith('win'):
    # this ensures OT dll is loaded
    import openturns

from .otECLM import ECLM

__version__ = '0.1'

