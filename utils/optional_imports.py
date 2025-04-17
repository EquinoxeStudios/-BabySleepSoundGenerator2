"""
Optional imports management and availability flags.
"""

import logging

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

# Try to import optional libraries - create fallbacks if not available
try:
    import noise
    HAS_PERLIN = True
except ImportError:
    HAS_PERLIN = False
    logger.warning(
        "Warning: 'noise' library not found. Installing it will enable higher quality Perlin noise textures."
    )
    logger.warning("You can install it with: pip install noise")

try:
    import pyloudnorm as pyln
    HAS_LOUDNORM = True
except ImportError:
    HAS_LOUDNORM = False
    logger.warning(
        "Warning: 'pyloudnorm' library not found. Installing it will enable EBU R128 loudness normalization."
    )
    logger.warning("You can install it with: pip install pyloudnorm")

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False
    logger.warning(
        "Warning: 'soundfile' library not found. It provides better audio file handling."
    )
    logger.warning("You can install it with: pip install soundfile")

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    logger.warning(
        "Warning: 'librosa' library not found. It provides advanced audio processing capabilities."
    )
    logger.warning("You can install it with: pip install librosa")

try:
    import pyroomacoustics as pra
    HAS_PYROOMACOUSTICS = True
except ImportError:
    HAS_PYROOMACOUSTICS = False
    logger.warning(
        "Warning: 'pyroomacoustics' library not found. It provides room impulse response simulations."
    )
    logger.warning("You can install it with: pip install pyroomacoustics")

try:
    from pydub import AudioSegment
    HAS_PYDUB = True
except ImportError:
    HAS_PYDUB = False
    logger.warning(
        "Warning: 'pydub' library not found. MP3 export will not be available."
    )
    logger.warning("You can install it with: pip install pydub")

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning(
        "Warning: 'matplotlib' library not found. Visualization will not be available."
    )
    logger.warning("You can install it with: pip install matplotlib")

try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    logger.warning(
        "Warning: 'numba' library not found. Performance optimizations will be limited."
    )
    logger.warning("You can install it with: pip install numba")