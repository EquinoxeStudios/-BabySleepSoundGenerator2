"""
Utility functions for BabySleepSoundGenerator.
"""

# Import only the necessary symbols from optional_imports
# Don't import from perlin_utils to avoid circular dependency
from utils.optional_imports import (
    HAS_PERLIN, 
    HAS_LOUDNORM, 
    HAS_SOUNDFILE, 
    HAS_LIBROSA, 
    HAS_PYROOMACOUSTICS, 
    HAS_PYDUB, 
    HAS_MATPLOTLIB, 
    HAS_NUMBA
)

# The functions from perlin_utils will be imported directly in the files that need them