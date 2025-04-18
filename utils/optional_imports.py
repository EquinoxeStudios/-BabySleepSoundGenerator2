"""
Optional imports management and availability flags.
"""

import logging
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger("BabySleepSoundGenerator")

# Initialize flags for optional dependencies
HAS_PERLIN = False
HAS_LOUDNORM = False
HAS_SOUNDFILE = False
HAS_LIBROSA = False
HAS_PYROOMACOUSTICS = False
HAS_PYDUB = False
HAS_MATPLOTLIB = False
HAS_NUMBA = False
HAS_TQDM = False

# Initialize module holders for optional dependencies
NOISE_MODULE = None
PYLOUDNORM_MODULE = None
SOUNDFILE_MODULE = None
LIBROSA_MODULE = None
PYROOMACOUSTICS_MODULE = None
PYDUB_AUDIOSEGMENT = None
MATPLOTLIB_PLT = None
NUMBA_MODULE = None
TQDM_MODULE = None

def check_imports() -> Dict[str, bool]:
    """Check which optional dependencies are available and return status."""
    global HAS_PERLIN, HAS_LOUDNORM, HAS_SOUNDFILE, HAS_LIBROSA
    global HAS_PYROOMACOUSTICS, HAS_PYDUB, HAS_MATPLOTLIB, HAS_NUMBA, HAS_TQDM
    global NOISE_MODULE, PYLOUDNORM_MODULE, SOUNDFILE_MODULE, LIBROSA_MODULE
    global PYROOMACOUSTICS_MODULE, PYDUB_AUDIOSEGMENT, MATPLOTLIB_PLT, NUMBA_MODULE, TQDM_MODULE

    # Try to import optional libraries - create fallbacks if not available
    try:
        import noise
        HAS_PERLIN = True
        NOISE_MODULE = noise
    except ImportError:
        HAS_PERLIN = False
        logger.warning(
            "Warning: 'noise' library not found. Installing it will enable higher quality Perlin noise textures."
        )
        logger.warning("You can install it with: pip install noise")

    try:
        import pyloudnorm as pyln
        HAS_LOUDNORM = True
        PYLOUDNORM_MODULE = pyln
    except ImportError:
        HAS_LOUDNORM = False
        logger.warning(
            "Warning: 'pyloudnorm' library not found. Installing it will enable EBU R128 loudness normalization."
        )
        logger.warning("You can install it with: pip install pyloudnorm")

    try:
        import soundfile as sf
        HAS_SOUNDFILE = True
        SOUNDFILE_MODULE = sf
    except ImportError:
        HAS_SOUNDFILE = False
        logger.warning(
            "Warning: 'soundfile' library not found. It provides better audio file handling."
        )
        logger.warning("You can install it with: pip install soundfile")

    try:
        import librosa
        HAS_LIBROSA = True
        LIBROSA_MODULE = librosa
    except ImportError:
        HAS_LIBROSA = False
        logger.warning(
            "Warning: 'librosa' library not found. It provides advanced audio processing capabilities."
        )
        logger.warning("You can install it with: pip install librosa")

    try:
        import pyroomacoustics as pra
        HAS_PYROOMACOUSTICS = True
        PYROOMACOUSTICS_MODULE = pra
    except ImportError:
        HAS_PYROOMACOUSTICS = False
        logger.warning(
            "Warning: 'pyroomacoustics' library not found. It provides room impulse response simulations."
        )
        logger.warning("You can install it with: pip install pyroomacoustics")

    try:
        from pydub import AudioSegment
        HAS_PYDUB = True
        PYDUB_AUDIOSEGMENT = AudioSegment
    except ImportError:
        HAS_PYDUB = False
        logger.warning(
            "Warning: 'pydub' library not found. MP3 export will not be available."
        )
        logger.warning("You can install it with: pip install pydub")

    try:
        import matplotlib.pyplot as plt
        HAS_MATPLOTLIB = True
        MATPLOTLIB_PLT = plt
    except ImportError:
        HAS_MATPLOTLIB = False
        logger.warning(
            "Warning: 'matplotlib' library not found. Visualization will not be available."
        )
        logger.warning("You can install it with: pip install matplotlib")

    try:
        import numba
        HAS_NUMBA = True
        NUMBA_MODULE = numba
    except ImportError:
        HAS_NUMBA = False
        logger.warning(
            "Warning: 'numba' library not found. Performance optimizations will be limited."
        )
        logger.warning("You can install it with: pip install numba")
        
    try:
        import tqdm
        HAS_TQDM = True
        TQDM_MODULE = tqdm
    except ImportError:
        HAS_TQDM = False
        logger.warning(
            "Warning: 'tqdm' library not found. Progress bars will not be available."
        )
        logger.warning("You can install it with: pip install tqdm")
        
    # Return dictionary of import statuses
    return {
        "perlin": HAS_PERLIN,
        "loudnorm": HAS_LOUDNORM,
        "soundfile": HAS_SOUNDFILE,
        "librosa": HAS_LIBROSA,
        "pyroomacoustics": HAS_PYROOMACOUSTICS,
        "pydub": HAS_PYDUB,
        "matplotlib": HAS_MATPLOTLIB,
        "numba": HAS_NUMBA,
        "tqdm": HAS_TQDM
    }

# Run the import checks when module is loaded
IMPORT_STATUS = check_imports()

def get_noise_module():
    """Get the noise module if available."""
    return NOISE_MODULE

def get_pyloudnorm_module():
    """Get the pyloudnorm module if available."""
    return PYLOUDNORM_MODULE

def get_soundfile_module():
    """Get the soundfile module if available."""
    return SOUNDFILE_MODULE

def get_librosa_module():
    """Get the librosa module if available."""
    return LIBROSA_MODULE

def get_pyroomacoustics_module():
    """Get the pyroomacoustics module if available."""
    return PYROOMACOUSTICS_MODULE

def get_pydub_audiosegment():
    """Get the pydub AudioSegment class if available."""
    return PYDUB_AUDIOSEGMENT

def get_matplotlib_plt():
    """Get the matplotlib.pyplot module if available."""
    return MATPLOTLIB_PLT

def get_numba_module():
    """Get the numba module if available."""
    return NUMBA_MODULE

def get_tqdm_module():
    """Get the tqdm module if available."""
    return TQDM_MODULE