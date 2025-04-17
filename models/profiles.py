"""
Predefined problem profiles for the BabySleepSoundGenerator.
"""

from typing import Dict

from models.constants import FrequencyFocus, RoomSize
from models.parameters import (
    HeartbeatParameters, FrequencyEmphasis, LowPassFilter, FrequencyLimiting, CircadianAlignment,
    DynamicVolume, MotionSmoothing, MoroReflexPrevention, SleepCycleModulation, 
    DynamicShushing, BreathingModulation, SafetyFeatures, ParentalVoice, ProblemProfile
)


def create_problem_profiles() -> Dict[str, ProblemProfile]:
    """Create and return predefined problem profiles"""
    
    problem_profiles = {
        "newborn_transition": ProblemProfile(
            primary_noise="womb",
            overlay_sounds=["heartbeat", "umbilical_swish"],
            frequency_focus=FrequencyFocus.LOW,
            recommended_volume=0.6,
            min_duration_hours=8,
            spatial_width=0.3,
            room_size=RoomSize.SMALL,
            description="Recreates womb-like environment for newborns adjusting to outside world",
            dynamic_volume=DynamicVolume(
                enabled=True,
                initial_db=66.0,
                reduction_db=50.0,
                reduction_time_minutes=15.0,
                fade_duration_seconds=180.0,
            ),
            heartbeat_parameters=HeartbeatParameters(
                variable_rate=True,
                base_bpm=70.0,
                bpm_range=(60.0, 80.0),
                variation_period_minutes=5.0,
            ),
            low_pass_filter=LowPassFilter(
                enabled=True,
                cutoff_hz=1000.0,
            ),
        ),
        "startle_reflex": ProblemProfile(
            primary_noise="pink",
            overlay_sounds=["heartbeat"],
            frequency_focus=FrequencyFocus.LOW_MID,
            recommended_volume=0.55,
            min_duration_hours=8,
            spatial_width=0.4,
            room_size=RoomSize.MEDIUM,
            description="Consistent sound to prevent startle reflex during sleep",
            motion_smoothing=MotionSmoothing(
                enabled=True,
                transition_seconds=5.0,
            ),
            moro_reflex_prevention=MoroReflexPrevention(
                enabled=True,
                burst_frequency_hz=45.0,
                burst_duration_seconds=0.5,
                interval_minutes=8.0,
            ),
            sleep_cycle_modulation=SleepCycleModulation(
                enabled=True,
                cycle_minutes=20.0,
            ),
            frequency_emphasis=FrequencyEmphasis(
                enabled=True,
                center_hz=500.0,
                bandwidth_hz=200.0,
                gain_db=3.0,
            ),
        ),
        "colic_relief": ProblemProfile(
            primary_noise="white",
            overlay_sounds=["shushing"],
            frequency_focus=FrequencyFocus.MID,
            recommended_volume=0.65,
            min_duration_hours=4,
            spatial_width=0.5,
            room_size=RoomSize.MEDIUM,
            description="Louder, more intense sound profile to help with colic episodes",
            dynamic_shushing=DynamicShushing(
                enabled=True,
                base_level_db=65.0,
                cry_response_db=3.0,
                response_time_ms=50.0,
            ),
            breathing_modulation=BreathingModulation(
                enabled=True,
                cycles_per_minute=12.0,
            ),
            safety_features=SafetyFeatures(
                auto_shutoff_minutes=45.0,
                high_volume_threshold_db=65.0,
            ),
            frequency_limiting=FrequencyLimiting(
                enabled=True,
                lower_limit_hz=200.0,
                upper_limit_hz=5000.0,
            ),
        ),
        "sleep_regression_4m": ProblemProfile(
            primary_noise="pink",
            overlay_sounds=[],
            frequency_focus=FrequencyFocus.BALANCED,
            recommended_volume=0.5,
            min_duration_hours=10,
            spatial_width=0.5,
            room_size=RoomSize.MEDIUM,
            description="Balanced sound profile to help with 4-month sleep regression",
            circadian_alignment=CircadianAlignment(
                enabled=True,
                evening_attenuation=True,
                high_freq_reduction_db=6.0,
                transition_duration_minutes=20.0,
            ),
            sleep_cycle_modulation=SleepCycleModulation(
                enabled=True,
                cycle_minutes=30.0,
            ),
            parental_voice=ParentalVoice(
                enabled=False,
                mix_level_db=-20.0,
            ),
        ),
        "teething_discomfort": ProblemProfile(
            primary_noise="white",
            overlay_sounds=[],
            frequency_focus=FrequencyFocus.MID_HIGH,
            recommended_volume=0.6,
            min_duration_hours=8,
            spatial_width=0.6,
            room_size=RoomSize.LARGE,
            description="Distracting sound profile to help with teething pain during sleep",
        ),
        "toddler_resistance": ProblemProfile(
            primary_noise="brown",
            overlay_sounds=[],
            frequency_focus=FrequencyFocus.LOW,
            recommended_volume=0.5,
            min_duration_hours=10,
            spatial_width=0.7,
            room_size=RoomSize.LARGE,
            description="Calming low-frequency sound to help toddlers wind down",
        ),
    }
    
    return problem_profiles