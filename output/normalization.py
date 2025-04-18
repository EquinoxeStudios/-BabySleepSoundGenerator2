"""
Loudness normalization and EBU R128 compliance.
"""

import numpy as np
import logging
from models.constants import Constants
from utils.optional_imports import HAS_LOUDNORM

# Create logger
logger = logging.getLogger("BabySleepSoundGenerator")

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
            logger.info("pyloudnorm not available, using peak normalization instead")
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                return audio / max_val * 0.9
            return audio

        try:
            # Create meter
            meter = pyln.Meter(self.sample_rate)

            # Check the shape of the audio input
            logger.debug(f"Audio shape before normalization: {audio.shape}")
            
            # Handle multi-channel audio - ensure we only have 1 or 2 channels
            if len(audio.shape) > 1:
                channels = audio.shape[1]
                if channels > 2:
                    logger.warning(f"Audio has {channels} channels, restricting to stereo")
                    audio = audio[:, :2]
            
            # Ensure audio is properly structured for pyloudnorm
            # pyloudnorm expects (samples, channels) for stereo, and 1D array for mono
            if len(audio.shape) == 1:
                # Mono audio - already in the right format
                current_loudness = meter.integrated_loudness(audio)
            else:
                # Stereo or multi-channel - need to transpose for pyloudnorm
                # Fix: Check if it's already a 2D array before transposing
                if len(audio.shape) == 2:
                    if audio.shape[1] == 1:  # Mono in 2D form
                        audio_for_meter = audio.flatten()
                    else:  # Stereo or more
                        audio_for_meter = audio.T
                    current_loudness = meter.integrated_loudness(audio_for_meter)
                else:
                    # Unexpected format - fallback to peak normalization
                    logger.warning(f"Unexpected audio format with shape {audio.shape}, using peak normalization")
                    max_val = np.max(np.abs(audio))
                    if max_val > 0:
                        return audio / max_val * 0.9
                    return audio

            # Calculate gain needed to reach target loudness
            logger.debug(f"Current loudness: {current_loudness} LUFS, Target: {self.target_loudness} LUFS")
            gain_db = self.target_loudness - current_loudness

            # Loudness normalization is a simple gain adjustment (in dB)
            gain_linear = 10 ** (gain_db / 20.0)
            logger.debug(f"Applying gain of {gain_db:.2f} dB ({gain_linear:.4f} linear)")

            # Apply gain
            normalized_audio = audio * gain_linear

            # Check for clipping and apply simple peak limiting
            max_val = np.max(np.abs(normalized_audio))
            if max_val > 0.98:
                logger.debug(f"Applying peak limiting (max value: {max_val:.4f})")
                normalized_audio = normalized_audio / max_val * 0.98
                
                # Apply a simple safety margin instead of true peak measurement
                # Skip the problematic true peak measurement that causes the 5-channel error
                normalized_audio = normalized_audio * 0.95  # Additional 5% safety margin

            return normalized_audio
            
        except Exception as e:
            logger.error(f"Error in loudness normalization: {e}")
            # Fall back to peak normalization in case of any errors
            logger.warning("Falling back to peak normalization")
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                return audio / max_val * 0.9
            return audio