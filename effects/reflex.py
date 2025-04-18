"""
Moro reflex prevention and other reflex-related effects.
"""

import numpy as np
from models.parameters import MoroReflexPrevention
from models.constants import Constants
from utils.perlin_utils import apply_modulation


class ReflexPreventer:
    """Applies techniques to help prevent Moro/startle reflex during sleep."""
    
    def __init__(self, sample_rate: int):
        """
        Initialize the reflex preventer.
        
        Args:
            sample_rate: Audio sample rate
        """
        self.sample_rate = sample_rate
        
    def apply_moro_reflex_prevention(
        self, audio: np.ndarray, params: MoroReflexPrevention
    ) -> np.ndarray:
        """
        Add specific low-frequency bursts to help prevent startle reflex.
        Based on research showing that specific frequencies can trigger the calming reflex.

        Args:
            audio: Input audio array
            params: Parameters for the Moro reflex prevention

        Returns:
            Modified audio with added calming bursts
        """
        if not params or not params.enabled:
            return audio

        # Extract parameters
        burst_freq = params.burst_frequency_hz
        burst_duration = params.burst_duration_seconds
        interval_minutes = params.interval_minutes

        # Create output array (same shape as input)
        output = audio.copy()

        # Check if input is stereo
        is_stereo = len(audio.shape) > 1

        # Convert interval to samples
        interval_samples = int(interval_minutes * 60 * self.sample_rate)
        burst_samples = int(burst_duration * self.sample_rate)

        # Calculate how many bursts to add
        num_bursts = len(audio) // interval_samples

        # Generate a single burst
        t_burst = np.linspace(0, burst_duration, burst_samples, endpoint=False)
        burst = 0.2 * np.sin(2 * np.pi * burst_freq * t_burst)  # 0.2 amplitude

        # Apply envelope to smooth the burst
        envelope = np.sin(np.linspace(0, np.pi, burst_samples)) ** 2  # Smooth rise and fall
        burst = burst * envelope

        # Add bursts at regular intervals efficiently
        for i in range(num_bursts):
            start_idx = i * interval_samples
            end_idx = min(start_idx + burst_samples, len(audio))
            burst_len = end_idx - start_idx

            if is_stereo:
                # For stereo, apply to both channels
                output[start_idx:end_idx] += burst[:burst_len, np.newaxis]
            else:
                output[start_idx:end_idx] += burst[:burst_len]

        # Normalize if needed
        max_val = np.max(np.abs(output))
        if max_val > Constants.MAX_AUDIO_VALUE:
            output = output / max_val * Constants.MAX_AUDIO_VALUE

        return output