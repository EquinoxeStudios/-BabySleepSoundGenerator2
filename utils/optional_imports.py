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

# Initialize flags for new enhanced dependencies
HAS_CUPY = False
HAS_TORCH = False
HAS_TORCHAUDIO = False
HAS_ISO226 = False
HAS_AUDIO_DIFFUSION = False

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

# Initialize module holders for new enhanced dependencies
CUPY_MODULE = None
TORCH_MODULE = None
TORCHAUDIO_MODULE = None
ISO226_MODULE = None
AUDIO_DIFFUSION_MODULE = None

def check_imports() -> Dict[str, bool]:
    """Check which optional dependencies are available and return status."""
    global HAS_PERLIN, HAS_LOUDNORM, HAS_SOUNDFILE, HAS_LIBROSA
    global HAS_PYROOMACOUSTICS, HAS_PYDUB, HAS_MATPLOTLIB, HAS_NUMBA, HAS_TQDM
    global HAS_CUPY, HAS_TORCH, HAS_TORCHAUDIO, HAS_ISO226, HAS_AUDIO_DIFFUSION
    global NOISE_MODULE, PYLOUDNORM_MODULE, SOUNDFILE_MODULE, LIBROSA_MODULE
    global PYROOMACOUSTICS_MODULE, PYDUB_AUDIOSEGMENT, MATPLOTLIB_PLT, NUMBA_MODULE, TQDM_MODULE
    global CUPY_MODULE, TORCH_MODULE, TORCHAUDIO_MODULE, ISO226_MODULE, AUDIO_DIFFUSION_MODULE

    # Try to import optional libraries - create fallbacks if not available
    try:
        import noise
        HAS_PERLIN = True
        NOISE_MODULE = noise
    except ImportError:
        HAS_PERLIN = False
        logger.error(
            "Error: Required dependency 'noise' not found. Please install it with: pip install noise"
        )

    try:
        import pyloudnorm as pyln
        HAS_LOUDNORM = True
        PYLOUDNORM_MODULE = pyln
    except ImportError:
        HAS_LOUDNORM = False
        logger.error(
            "Error: Required dependency 'pyloudnorm' not found. Please install it with: pip install pyloudnorm"
        )

    try:
        import soundfile as sf
        HAS_SOUNDFILE = True
        SOUNDFILE_MODULE = sf
    except ImportError:
        HAS_SOUNDFILE = False
        logger.error(
            "Error: Required dependency 'soundfile' not found. Please install it with: pip install soundfile"
        )

    try:
        import librosa
        HAS_LIBROSA = True
        LIBROSA_MODULE = librosa
    except ImportError:
        HAS_LIBROSA = False
        logger.error(
            "Error: Required dependency 'librosa' not found. Please install it with: pip install librosa"
        )

    try:
        import pyroomacoustics as pra
        HAS_PYROOMACOUSTICS = True
        PYROOMACOUSTICS_MODULE = pra
    except ImportError:
        HAS_PYROOMACOUSTICS = False
        logger.error(
            "Error: Required dependency 'pyroomacoustics' not found. Please install it with: pip install pyroomacoustics"
        )

    try:
        from pydub import AudioSegment
        HAS_PYDUB = True
        PYDUB_AUDIOSEGMENT = AudioSegment
    except ImportError:
        HAS_PYDUB = False
        logger.error(
            "Error: Required dependency 'pydub' not found. Please install it with: pip install pydub"
        )

    try:
        import matplotlib.pyplot as plt
        HAS_MATPLOTLIB = True
        MATPLOTLIB_PLT = plt
    except ImportError:
        HAS_MATPLOTLIB = False
        logger.error(
            "Error: Required dependency 'matplotlib' not found. Please install it with: pip install matplotlib"
        )

    try:
        import numba
        HAS_NUMBA = True
        NUMBA_MODULE = numba
    except ImportError:
        HAS_NUMBA = False
        logger.error(
            "Error: Required dependency 'numba' not found. Please install it with: pip install numba"
        )
        
    try:
        import tqdm
        HAS_TQDM = True
        TQDM_MODULE = tqdm
    except ImportError:
        HAS_TQDM = False
        logger.error(
            "Error: Required dependency 'tqdm' not found. Please install it with: pip install tqdm"
        )
    
    # Check for enhanced dependencies
    try:
        import cupy
        HAS_CUPY = True
        CUPY_MODULE = cupy
        logger.info("CuPy found - GPU acceleration available")
    except ImportError:
        HAS_CUPY = False
        logger.error("Error: Required dependency 'cupy' not found. Please install it with: pip install cupy")
        
    try:
        import torch
        HAS_TORCH = True
        TORCH_MODULE = torch
        logger.info("PyTorch found")
    except ImportError:
        HAS_TORCH = False
        logger.error("Error: Required dependency 'torch' not found. Please install it with: pip install torch")
        
    try:
        import torchaudio
        HAS_TORCHAUDIO = True
        TORCHAUDIO_MODULE = torchaudio
        logger.info(f"TorchAudio found: version {torchaudio.__version__}")
    except ImportError:
        HAS_TORCHAUDIO = False
        logger.error("Error: Required dependency 'torchaudio' not found. Please install it with: pip install torchaudio")
        
    try:
        import iso226
        HAS_ISO226 = True
        ISO226_MODULE = iso226
        logger.info("ISO-226 library found")
    except ImportError:
        HAS_ISO226 = False
        logger.error("Error: Required dependency 'iso226' not found. Please install it with: pip install iso226")
        
    try:
        import audio_diffusion_pytorch
        HAS_AUDIO_DIFFUSION = True
        AUDIO_DIFFUSION_MODULE = audio_diffusion_pytorch
        logger.info("Audio Diffusion library found")
    except ImportError:
        HAS_AUDIO_DIFFUSION = False
        logger.error("Error: Required dependency 'audio-diffusion-pytorch' not found. Please install it with: pip install audio-diffusion-pytorch")
        
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
        "tqdm": HAS_TQDM,
        "cupy": HAS_CUPY,
        "torch": HAS_TORCH,
        "torchaudio": HAS_TORCHAUDIO,
        "iso226": HAS_ISO226,
        "audio_diffusion": HAS_AUDIO_DIFFUSION
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

def get_cupy_module():
    """Get the cupy module if available."""
    return CUPY_MODULE

def get_torch_module():
    """Get the PyTorch module if available."""
    return TORCH_MODULE

def get_torchaudio_module():
    """Get the TorchAudio module if available."""
    return TORCHAUDIO_MODULE

def get_iso226_module():
    """Get the ISO-226 module if available."""
    return ISO226_MODULE

def get_audio_diffusion_module():
    """Get the audio diffusion module if available."""
    return AUDIO_DIFFUSION_MODULE