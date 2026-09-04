"""
Backward-compatibility proxy for ai_dna.data.
All data loading, processing, and streaming functionality has been modularized into `ai_dna.data`.
Direct imports like `from data import ...` or `import data` continue to work without modification.
"""

from ai_dna.data import *
