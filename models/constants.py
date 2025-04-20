"""
Constants and enumerations for the BabySleepSoundGenerator.
"""

from enum import Enum


class Constants:
    """General constants used throughout the application"""
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
    
    # Looping parameters - added these new constants
    LOOP_SEARCH_START_PERCENTAGE = 0.7  # Start searching at 70% of the file
    LOOP_SEARCH_STEP_DIVIDER = 20  # Search step size divisor


class HeartbeatConstants:
    """Constants for heartbeat sound generation"""
    LUB_DURATION_SECONDS = 0.12
    DUB_DURATION_SECONDS = 0.10
    DUB_DELAY_SECONDS = 0.20
    AMPLITUDE_VARIATION_PCT = 0.15  # 15% amplitude variation
    LUB_FREQUENCY_HZ = 60.0
    DUB_FREQUENCY_HZ = 45.0
    DEFAULT_AMPLITUDE = 0.8


class ShushingConstants:
    """Constants for shushing sound generation"""
    BANDPASS_LOW_HZ = 2000
    BANDPASS_HIGH_HZ = 4000
    SHUSH_RATE_PER_SECOND = 1.5
    MODULATION_MIN = 0.3
    MODULATION_MAX = 1.0
    DEFAULT_AMPLITUDE = 0.9


class FanConstants:
    """Constants for fan sound generation"""
    BASE_ROTATION_HZ = 4.0
    SPEED_VARIATION_PCT = 0.1  # 10% speed variation
    RESONANCE_FREQUENCIES_HZ = [180, 320, 560, 820]
    RESONANCE_Q_FACTOR = 10.0
    BANDPASS_LOW_HZ = 80
    BANDPASS_HIGH_HZ = 4000
    DEFAULT_AMPLITUDE = 0.85


class WombConstants:
    """Constants for womb sound generation"""
    BREATHING_RATE_BREATHS_PER_MIN = 16
    BLOOD_FLOW_CYCLES_PER_MIN = 4
    BREATHING_MODULATION_DEPTH = 0.15
    DEEP_RHYTHM_MODULATION_DEPTH = 0.1
    WOMB_LOWPASS_CUTOFF_HZ = 1000
    DEFAULT_AMPLITUDE = 0.9


class UmbilicalConstants:
    """Constants for umbilical swish sound generation"""
    WHOOSH_BANDPASS_LOW_HZ = 50
    WHOOSH_BANDPASS_HIGH_HZ = 150
    PULSE_RATE_BPM = 70
    PULSE_FACTOR_BASE = 0.5
    PULSE_FACTOR_VARIATION = 0.25
    SECONDARY_BANDPASS_LOW_HZ = 100
    SECONDARY_BANDPASS_HIGH_HZ = 250
    DEFAULT_AMPLITUDE = 0.9


class PerformanceConstants:
    """Constants for performance tuning"""
    DEFAULT_BUFFER_SIZE = 8192
    PERLIN_STRETCH_FACTOR = 100
    FFT_CHUNK_SIZE_SECONDS = 60  # Process 1 minute chunks for FFT operations
    MAX_DURATION_SECONDS_BEFORE_CHUNKING = 300  # 5 minutes threshold
    CROSSFADE_BETWEEN_CHUNKS_SECONDS = 1.0


class SpatialConstants:
    """Constants for spatial processing"""
    MAX_STEREO_DELAY_MS = 0.5  # Maximum stereo delay in milliseconds
    HRTF_IMPULSE_LENGTH = 512  # Length of HRTF impulse response in samples
    DECORRELATION_NOISE_DURATION_MS = 10  # 10ms noise for decorrelation
    DEFAULT_STEREO_WIDTH = 0.5


class NoiseEnhancementConstants:
    """Constants for enhanced noise generation features"""
    DEFAULT_EQUAL_LOUDNESS_LEVEL = 50  # ISO-226 equal loudness curve at 50 dB
    DEFAULT_LIMITER_THRESHOLD_DB = -10.0
    DEFAULT_LIMITER_KNEE_DB = 6.0
    DEFAULT_LIMITER_ATTACK_MS = 5.0
    DEFAULT_LIMITER_RELEASE_MS = 50.0
    DEFAULT_LIMITER_LOOKAHEAD_MS = 10.0
    DEFAULT_ORGANIC_DRIFT_BANDS = 10
    DEFAULT_ORGANIC_DRIFT_MIN_FREQ = 20
    DEFAULT_ORGANIC_DRIFT_MAX_FREQ = 20000
    DEFAULT_DIFFUSION_STRENGTH = 0.5
    DEFAULT_FADE_IN_SECONDS = 10.0
    DEFAULT_FADE_OUT_SECONDS = 60.0


class NoiseColor(str, Enum):
    """Enumeration of noise colors"""
    WHITE = "white"
    PINK = "pink"
    BROWN = "brown"


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