"""
Instinct Extraction and Slow Clock Encoding Package.
"""

from .cppn_encoder import InverseCPPNEncoder
from .ewc import EWCConsolidator
from .slow_clock import SlowClockEncoder

__all__ = [
    "InverseCPPNEncoder",
    "EWCConsolidator",
    "SlowClockEncoder",
]
