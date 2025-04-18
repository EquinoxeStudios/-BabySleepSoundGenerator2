"""
Frequency shaping and filtering operations.
"""

import numpy as np
from scipy import signal
from typing import Dict, Any, Optional, Union

from models.constants import FrequencyFocus
from models.parameters import FrequencyEmphasis, LowPassFilter, FrequencyLimiting, CircadianAlignment


class FrequencyProcessor:
    """Processor for frequency shaping and filtering operations."""
    
    def __init__(self, sample_rate: int):
        """
        Initialize the frequency processor.
        
        Args:
            sample_rate: Audio sample rate in Hz
        """
        self.sample_rate = sample_rate
        
    def apply_frequency_shaping(
        self,
        audio: np.ndarray,
        focus: Union[FrequencyFocus, str],
        additional_parameters: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """
        Apply enhanced frequency shaping based on desired focus and optional additional parameters.

        Args:
            audio: Input audio signal
            focus: Type of frequency shaping to apply (enum or string)
            additional_parameters: Optional parameters for custom frequency shaping

        Returns:
            Frequency-shaped audio
        """
        # Convert string to enum if needed
        if isinstance(focus, str):
            try:
                focus = FrequencyFocus(focus)
            except ValueError:
                # Fallback to balanced if invalid focus provided
                focus = FrequencyFocus.BALANCED
                
        # Apply any additional parameters first if provided
        processed_audio = audio.copy()
        additional_parameters = additional_parameters or {}

        # Apply frequency emphasis if requested (for Startle Reflex profile)
        if isinstance(additional_parameters.get("frequency_emphasis"), FrequencyEmphasis):
            emphasis = additional_parameters["frequency_emphasis"]
            if emphasis.enabled:
                try:
                    center_hz = emphasis.center_hz
                    bandwidth_hz = emphasis.bandwidth_hz
                    gain_db = emphasis.gain_db

                    # Create a bandpass filter centered on the target frequency
                    lower_freq = max(20, center_hz - bandwidth_hz / 2)
                    upper_freq = min(20000, center_hz + bandwidth_hz / 2)

                    # Normalize to Nyquist frequency
                    nyquist = self.sample_rate / 2
                    lower_norm = lower_freq / nyquist
                    upper_norm = upper_freq / nyquist

                    # Apply bandpass filter
                    b, a = signal.butter(4, [lower_norm, upper_norm], "band")
                    filtered = signal.lfilter(b, a, processed_audio)

                    # Apply gain (convert dB to linear)
                    gain_linear = 10 ** (gain_db / 20.0)

                    # Mix with original
                    processed_audio = processed_audio + (filtered * (gain_linear - 1.0))
                except Exception as e:
                    import logging
                    logger = logging.getLogger("BabySleepSoundGenerator")
                    logger.error(f"Error applying frequency emphasis: {e}")

        # Apply low-pass filter if requested (for Newborn Transition profile)
        if isinstance(additional_parameters.get("low_pass_filter"), LowPassFilter):
            lpf = additional_parameters["low_pass_filter"]
            if lpf.enabled:
                try:
                    cutoff_hz = lpf.cutoff_hz

                    # Normalize to Nyquist frequency
                    nyquist = self.sample_rate / 2
                    cutoff_norm = cutoff_hz / nyquist

                    # Higher order filter (6th) for sharper cutoff
                    b, a = signal.butter(6, cutoff_norm, "low")
                    processed_audio = signal.lfilter(b, a, processed_audio)
                except Exception as e:
                    import logging
                    logger = logging.getLogger("BabySleepSoundGenerator")
                    logger.error(f"Error applying low-pass filter: {e}")

        # Apply frequency limiting if requested (for Colic Relief profile)
        if isinstance(additional_parameters.get("frequency_limiting"), FrequencyLimiting):
            freq_limit = additional_parameters["frequency_limiting"]
            if freq_limit.enabled:
                try:
                    lower_limit_hz = freq_limit.lower_limit_hz
                    upper_limit_hz = freq_limit.upper_limit_hz

                    # Normalize to Nyquist frequency
                    nyquist = self.sample_rate / 2
                    lower_norm = lower_limit_hz / nyquist
                    upper_norm = upper_limit_hz / nyquist

                    # Apply bandpass filter
                    b, a = signal.butter(4, [lower_norm, upper_norm], "band")
                    processed_audio = signal.lfilter(b, a, processed_audio)
                except Exception as e:
                    import logging
                    logger = logging.getLogger("BabySleepSoundGenerator")
                    logger.error(f"Error applying frequency limiting: {e}")

        # Apply circadian alignment (reduced high frequencies) for evening (4-month sleep regression)
        if isinstance(additional_parameters.get("circadian_alignment"), CircadianAlignment):
            circadian = additional_parameters["circadian_alignment"]
            if circadian.enabled and circadian.evening_attenuation:
                try:
                    high_freq_reduction_db = circadian.high_freq_reduction_db

                    # Create a custom filter that gradually reduces high frequencies
                    freqs = [0, 1000, 2000, 4000, 8000, 16000, self.sample_rate // 2]

                    # Convert dB reduction to linear scale (gradually increasing reduction)
                    reduction_factor = 10 ** (-high_freq_reduction_db / 20.0)
                    gains = [
                        1.0, 1.0, 0.9, 0.7, reduction_factor, reduction_factor, reduction_factor
                    ]

                    # Normalize frequencies to Nyquist
                    norm_freqs = [f / (self.sample_rate / 2) for f in freqs]

                    # Create FIR filter for smooth response
                    evening_filter = signal.firwin2(101, norm_freqs, gains)
                    processed_audio = signal.lfilter(evening_filter, 1, processed_audio)
                except Exception as e:
                    import logging
                    logger = logging.getLogger("BabySleepSoundGenerator")
                    logger.error(f"Error applying circadian alignment: {e}")

        # Now apply the standard frequency shaping based on focus
        try:
            if focus == FrequencyFocus.LOW:
                # Emphasize low frequencies (for womb-like, deep sleep sounds)
                # Use higher order filter for steeper cutoff
                b, a = signal.butter(6, 1000 / (self.sample_rate / 2), "low")
                filtered = signal.lfilter(b, a, processed_audio)

                # Add subtle bass boost around 100-200Hz
                bass_boost = signal.firwin2(
                    101,
                    [0, 0.005, 0.01, 0.015, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0],
                    [1.0, 1.5, 1.8, 1.5, 1.2, 1.0, 0.8, 0.5, 0.2, 0.1],
                    fs=self.sample_rate,
                )
                return signal.lfilter(bass_boost, 1, filtered)

            elif focus == FrequencyFocus.LOW_MID:
                # Focus on low and mid frequencies with more refined curve
                # Create a custom filter with a smoother response curve
                freqs = [0, 200, 400, 800, 1600, 2400, 3200, 5000, 8000, self.sample_rate // 2]
                gains = [0.6, 1.0, 1.2, 1.0, 0.8, 0.5, 0.3, 0.2, 0.1, 0.1]

                # Normalize frequencies to Nyquist
                norm_freqs = [f / (self.sample_rate / 2) for f in freqs]

                # Create FIR filter
                fir_filter = signal.firwin2(101, norm_freqs, gains)
                return signal.lfilter(fir_filter, 1, processed_audio)

            elif focus == FrequencyFocus.MID:
                # Focus on mid frequencies with more precise control
                # More precise bandpass with better transition bands
                b, a = signal.butter(
                    6, [400 / (self.sample_rate / 2), 3200 / (self.sample_rate / 2)], "band"
                )
                filtered = signal.lfilter(b, a, processed_audio)

                # Add a slight boost around 1-2kHz (most sensitive hearing range)
                mid_boost = signal.firwin2(
                    101,
                    [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 1.0],
                    [0.3, 0.6, 0.9, 1.2, 1.0, 0.9, 0.7, 0.5, 0.3, 0.1],
                    fs=self.sample_rate,
                )
                return signal.lfilter(mid_boost, 1, filtered)

            elif focus == FrequencyFocus.MID_HIGH:
                # Focus on mid to high frequencies with better high-end extension
                # Create a custom filter with more precise high-frequency response
                freqs = [0, 500, 1000, 2000, 4000, 6000, 8000, 12000, self.sample_rate // 2]
                gains = [0.1, 0.3, 0.8, 1.2, 1.0, 0.9, 0.7, 0.5, 0.3]

                # Normalize frequencies to Nyquist
                norm_freqs = [f / (self.sample_rate / 2) for f in freqs]

                # Create FIR filter
                fir_filter = signal.firwin2(101, norm_freqs, gains)
                return signal.lfilter(fir_filter, 1, processed_audio)

            elif focus == FrequencyFocus.BALANCED:
                # Balanced frequency response with more natural curve
                # Create a custom filter that mimics pleasant "hi-fi" response
                freqs = [0, 30, 80, 200, 500, 1000, 2000, 4000, 8000, 12000, self.sample_rate // 2]
                gains = [0.5, 0.8, 1.0, 1.1, 1.0, 0.95, 1.0, 1.05, 0.9, 0.7, 0.5]

                # Normalize frequencies to Nyquist
                norm_freqs = [f / (self.sample_rate / 2) for f in freqs]

                # Create FIR filter
                fir_filter = signal.firwin2(101, norm_freqs, gains)
                return signal.lfilter(fir_filter, 1, processed_audio)

            else:
                # Default: return unmodified
                return processed_audio
        except Exception as e:
            import logging
            logger = logging.getLogger("BabySleepSoundGenerator")
            logger.error(f"Error applying frequency shaping: {e}")
            # Return original audio if shaping fails
            return audio