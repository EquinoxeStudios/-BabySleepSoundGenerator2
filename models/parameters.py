"""
Parameter dataclasses for the BabySleepSoundGenerator.
"""

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple, Any


@dataclass
class HeartbeatParameters:
    """Parameters for heartbeat sound generation"""
    variable_rate: bool = False
    base_bpm: float = 70.0
    bpm_range: Tuple[float, float] = (60.0, 80.0)
    variation_period_minutes: float = 5.0


@dataclass
class FrequencyEmphasis:
    """Parameters for frequency emphasis"""
    enabled: bool = False
    center_hz: float = 500.0
    bandwidth_hz: float = 200.0
    gain_db: float = 3.0


@dataclass
class LowPassFilter:
    """Parameters for low-pass filter"""
    enabled: bool = False
    cutoff_hz: float = 1000.0


@dataclass
class FrequencyLimiting:
    """Parameters for frequency limiting"""
    enabled: bool = False
    lower_limit_hz: float = 200.0
    upper_limit_hz: float = 5000.0


@dataclass
class CircadianAlignment:
    """Parameters for circadian rhythm alignment"""
    enabled: bool = False
    evening_attenuation: bool = False
    high_freq_reduction_db: float = 6.0
    transition_duration_minutes: float = 20.0


@dataclass
class DynamicVolume:
    """Parameters for dynamic volume adjustment"""
    enabled: bool = False
    initial_db: float = 66.0
    reduction_db: float = 50.0
    reduction_time_minutes: float = 15.0
    fade_duration_seconds: float = 180.0


@dataclass
class MotionSmoothing:
    """Parameters for motion smoothing"""
    enabled: bool = False
    transition_seconds: float = 5.0


@dataclass
class MoroReflexPrevention:
    """Parameters for Moro reflex prevention"""
    enabled: bool = False
    burst_frequency_hz: float = 45.0
    burst_duration_seconds: float = 0.5
    interval_minutes: float = 8.0


@dataclass
class SleepCycleModulation:
    """Parameters for sleep cycle modulation"""
    enabled: bool = False
    cycle_minutes: float = 20.0


@dataclass
class DynamicShushing:
    """Parameters for dynamic shushing"""
    enabled: bool = False
    base_level_db: float = 65.0
    cry_response_db: float = 3.0
    response_time_ms: float = 50.0


@dataclass
class BreathingModulation:
    """Parameters for breathing modulation"""
    enabled: bool = False
    cycles_per_minute: float = 12.0


@dataclass
class SafetyFeatures:
    """Parameters for safety features"""
    auto_shutoff_minutes: float = 45.0
    high_volume_threshold_db: float = 65.0


@dataclass
class ParentalVoice:
    """Parameters for parental voice overlay"""
    enabled: bool = False
    mix_level_db: float = -20.0


@dataclass
class ProblemProfile:
    """A complete profile for a specific baby sleep problem"""
    primary_noise: str
    overlay_sounds: List[str] = field(default_factory=list)
    frequency_focus: str = "balanced"
    recommended_volume: float = 0.5
    min_duration_hours: int = 8
    spatial_width: float = 0.5
    room_size: str = "medium"
    description: str = ""
    
    # Enhanced parameters
    dynamic_volume: Optional[DynamicVolume] = None
    heartbeat_parameters: Optional[HeartbeatParameters] = None
    low_pass_filter: Optional[LowPassFilter] = None
    motion_smoothing: Optional[MotionSmoothing] = None
    moro_reflex_prevention: Optional[MoroReflexPrevention] = None
    sleep_cycle_modulation: Optional[SleepCycleModulation] = None
    frequency_emphasis: Optional[FrequencyEmphasis] = None
    frequency_limiting: Optional[FrequencyLimiting] = None
    circadian_alignment: Optional[CircadianAlignment] = None
    dynamic_shushing: Optional[DynamicShushing] = None
    breathing_modulation: Optional[BreathingModulation] = None
    safety_features: Optional[SafetyFeatures] = None
    parental_voice: Optional[ParentalVoice] = None