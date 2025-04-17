"""
Constants and enumerations for the BabySleepSoundGenerator.
"""

from enum import Enum


class Constants:
    """Constants used throughout the application"""
    # Audio settings
    DEFAULT_SAMPLE_RATE = 48000
    DEFAULT_BIT_DEPTH = 24
    DEFAULT_CHANNELS = 2
    DEFAULT_TARGET_LOUDNESS = -23.0  # EBU R128 standard
    
    # Output value limits
    MAX_AUDIO_VALUE = 0.95  # Maximum absolute value for audio samples
    
    # Safety guidelines
    AAP_MAX_DB_SPL = 65.0  # American Academy of Pediatrics guideline
    WHO_SAFE_HOURS = {
        65.0: 8.0,  # 8 hours at 65 dB SPL
        70.0: 4.0,  # 4 hours at 70 dB SPL
        75.0: 2.0,  # 2 hours at 75 dB SPL
        80.0: 1.0,  # 1 hour at 80 dB SPL
    }
    
    # Processing parameters
    DEFAULT_CROSSFADE_DURATION = 15.0  # seconds
    BREATHING_RATE_CPM = 12.0  # cycles per minute
    DEFAULT_HEARTBEAT_BPM = 70.0


class FrequencyFocus(str, Enum):
    """Enumeration of frequency focus options"""
    LOW = "low"
    LOW_MID = "low_mid"
    MID = "mid"
    MID_HIGH = "mid_high"
    BALANCED = "balanced"


class RoomSize(str, Enum):
    """Enumeration of room size options"""
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


class LoopingMethod(str, Enum):
    """Enumeration of looping methods"""
    CROSS_CORRELATION = "cross_correlation"
    SIMPLE = "simple"


class OutputFormat(str, Enum):
    """Enumeration of output formats"""
    WAV = "wav"
    MP3 = "mp3"