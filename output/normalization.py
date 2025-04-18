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
            return self._apply_peak_normalization(audio)

        try:
            # Check the audio shape first before creating the meter
            logger.debug(f"Audio shape before normalization: {audio.shape}")
            
            # Handle multi-channel audio - ensure we only have 1 or 2 channels
            if len(audio.shape) > 1:
                channels = audio.shape[1]
                if channels > 2:
                    logger.warning(f"Audio has {channels} channels, restricting to stereo")
                    audio = audio[:, :2]
            
            # Check the shape again
            audio_shape = audio.shape
            
            # Pre-emptively check for conditions that will cause the five-channel error
            # and use peak normalization instead
            if len(audio_shape) > 1 and audio_shape[1] > 2:
                logger.warning(f"Audio has unexpected shape {audio_shape}, using peak normalization")
                return self._apply_peak_normalization(audio)

            # Check for problematic audio values before processing
            if np.isnan(audio).any() or np.isinf(audio).any():
                logger.warning("Audio contains NaN or Inf values, using peak normalization")
                # Clean the values first
                audio = np.nan_to_num(audio, nan=0.0, posinf=0.95, neginf=-0.95)
                return self._apply_peak_normalization(audio)
                
            # Create meter
            meter = pyln.Meter(self.sample_rate)

            # Ensure audio is properly structured for pyloudnorm
            # pyloudnorm expects (samples, channels) for stereo, and 1D array for mono
            if len(audio.shape) == 1:
                # Mono audio - already in the right format
                try:
                    current_loudness = meter.integrated_loudness(audio)
                except Exception as e:
                    logger.warning(f"Error measuring loudness for mono audio: {e}")
                    return self._apply_peak_normalization(audio)
            else:
                # Stereo or multi-channel - need to transpose for pyloudnorm
                try:
                    if audio.shape[1] == 1:  # Mono in 2D form
                        current_loudness = meter.integrated_loudness(audio.flatten())
                    elif audio.shape[1] == 2:  # Stereo
                        current_loudness = meter.integrated_loudness(audio.T)
                    else:
                        # More than 2 channels - fall back to peak normalization
                        logger.warning(f"Audio has {audio.shape[1]} channels, using peak normalization")
                        return self._apply_peak_normalization(audio)
                except Exception as e:
                    logger.warning(f"Error measuring loudness: {e}")
                    return self._apply_peak_normalization(audio)

            # Check if we got a valid loudness measurement
            if current_loudness is None or np.isnan(current_loudness) or np.isinf(current_loudness):
                logger.warning("Invalid loudness measurement, using peak normalization")
                return self._apply_peak_normalization(audio)

            # Calculate gain needed to reach target loudness
            logger.debug(f"Current loudness: {current_loudness} LUFS, Target: {self.target_loudness} LUFS")
            gain_db = self.target_loudness - current_loudness

            # Check for extreme gain values
            if abs(gain_db) > 40:  # If gain adjustment is extreme
                logger.warning(f"Extreme gain adjustment required ({gain_db:.1f} dB), using peak normalization instead")
                return self._apply_peak_normalization(audio)

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
                
            # Apply a simple safety margin to avoid potential clipping
            normalized_audio = normalized_audio * 0.95  # 5% safety margin

            return normalized_audio
            
        except Exception as e:
            logger.warning(f"Error in loudness normalization: {e}, using peak normalization")
            return self._apply_peak_normalization(audio)
    
    def _apply_peak_normalization(self, audio: np.ndarray, target_peak: float = 0.9) -> np.ndarray:
        """
        Apply simple peak normalization as a fallback method.
        
        Args:
            audio: Input audio array
            target_peak: Target peak level (0.0-1.0)
            
        Returns:
            Peak-normalized audio
        """
        try:
            # Clean up any NaN or Inf values first
            if np.isnan(audio).any() or np.isinf(audio).any():
                logger.debug("Cleaning NaN/Inf values before peak normalization")
                audio = np.nan_to_num(audio, nan=0.0, posinf=0.95, neginf=-0.95)
            
            # Find the peak value
            max_val = np.max(np.abs(audio))
            
            if max_val > 0:
                # Normalize to target peak
                return audio / max_val * target_peak
            else:
                # If audio is silent, return as is
                return audio
        except Exception as e:
            logger.error(f"Error in peak normalization: {e}")
            # Return original audio if everything fails
            return audio