"""
Utility functions for BabySleepSoundGenerator.
"""

# Import only the necessary symbols from optional_imports
from utils.optional_imports import (
    HAS_PERLIN, 
    HAS_LOUDNORM, 
    HAS_SOUNDFILE, 
    HAS_LIBROSA, 
    HAS_PYROOMACOUSTICS, 
    HAS_PYDUB, 
    HAS_MATPLOTLIB, 
    HAS_NUMBA,
    HAS_TQDM,
    get_noise_module,
    get_pyloudnorm_module,
    get_soundfile_module,
    get_librosa_module,
    get_pyroomacoustics_module,
    get_pydub_audiosegment,
    get_matplotlib_plt,
    get_numba_module,
    get_tqdm_module,
    check_imports
)

# Import perlin utilities functions to make them available through utils
from utils.perlin_utils import (
    generate_perlin_noise,
    apply_modulation,
    generate_dynamic_modulation
)

# Import common utility functions to make them available through utils
from utils.random_state import RandomStateManager
from utils.config import ConfigManager
from utils.logging import setup_logging

# Version information
__version__ = "1.0.0"