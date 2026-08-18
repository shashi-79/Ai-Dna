"""
Instinct Extraction and Slow Clock Encoding Package.
"""

from .svd_filter import SVDInstinctFilter
from .cppn_encoder import InverseCPPNEncoder
from .ewc import EWCConsolidator
from .slow_clock import SlowClockEncoder

__all__ = [
    "SVDInstinctFilter",
    "InverseCPPNEncoder",
    "EWCConsolidator",
    "SlowClockEncoder",
]
