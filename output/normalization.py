"""
Loudness normalization and EBU R128 compliance.
"""

import numpy as np
from models.constants import Constants
from utils.optional_imports import HAS_LOUDNORM

# Import optional libraries
if HAS_LOUDNORM:
    import pyloudnorm as pyln


class LoudnessNormalizer:
    """Handles EBU R128 / ITU-R BS.1770 loudness normalization."""
    
    def __init__(self, sample_rate: int, target_loudness: float = Constants.DEFAULT_TARGET_LOUDNESS):
        """
        Initialize the loudness normalizer.
        
        Args:
            sample_rate: Audio sample rate
            target_loudness: Target loudness in LUFS
        """
        self.sample_rate = sample_rate
        self.target_loudness = target_loudness
        
    def apply_ebu_r128_normalization(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply EBU R128 loudness normalization to the audio.
        This ensures broadcast standard loudness levels.

        Args:
            audio: Input audio array (mono or stereo)

        Returns:
            Loudness normalized audio
        """
        if not HAS_LOUDNORM:
            # If pyloudnorm is not available, apply simple peak normalization
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                return audio / max_val * 0.9
            return audio

        # Create meter
        meter = pyln.Meter(self.sample_rate)

        # Measure current loudness
        if len(audio.shape) == 1:
            # Mono
            current_loudness = meter.integrated_loudness(audio)
        else:
            # Stereo
            current_loudness = meter.integrated_loudness(audio.T)

        # Calculate gain needed to reach target loudness
        gain_db = self.target_loudness - current_loudness

        # Loudness normalization is a simple gain adjustment (in dB)
        gain_linear = 10 ** (gain_db / 20.0)

        # Apply gain
        normalized_audio = audio * gain_linear

        # Check for clipping and apply true-peak limiting if needed
        max_val = np.max(np.abs(normalized_audio))
        if max_val > 0.98:
            # Apply a true-peak limiter to prevent clipping
            normalized_audio = normalized_audio / max_val * 0.98

            # Add a second check using pyloudnorm's true peak measurement
            if len(audio.shape) == 1:
                true_peak = meter.true_peak(normalized_audio)
            else:
                true_peak = max(
                    meter.true_peak(normalized_audio[:, 0]),
                    meter.true_peak(normalized_audio[:, 1]),
                )

            if true_peak > -1.0:  # -1 dBTP limit per EBU R128
                normalized_audio = normalized_audio / (10 ** (true_peak / 20)) * 0.891  # -1 dBTP

        return normalized_audio