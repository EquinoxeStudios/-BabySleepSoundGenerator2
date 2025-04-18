"""
Dynamic volume adjustment effects.
"""

import numpy as np
from models.parameters import DynamicVolume, SafetyFeatures
from utils.perlin_utils import apply_modulation


class DynamicVolumeProcessor:
    """Processor for dynamic volume and auto-shutoff features."""
    
    def __init__(self, sample_rate: int, volume_to_db_spl: dict):
        """
        Initialize the dynamic volume processor.
        
        Args:
            sample_rate: Audio sample rate
            volume_to_db_spl: Dictionary mapping volume levels to dB SPL values
        """
        self.sample_rate = sample_rate
        self.volume_to_db_spl = volume_to_db_spl
        
    def apply_dynamic_volume(
        self, audio: np.ndarray, params: DynamicVolume
    ) -> np.ndarray:
        """
        Apply automatic volume reduction after a specified time.
        Important for preventing habituation and providing proper sleep support.

        Args:
            audio: Input audio array
            params: Parameters for dynamic volume

        Returns:
            Volume-adjusted audio
        """
        if not params or not params.enabled:
            return audio

        # Extract parameters
        initial_db = params.initial_db
        reduction_db = params.reduction_db
        reduction_time_minutes = params.reduction_time_minutes
        fade_duration_seconds = params.fade_duration_seconds

        # Find the closest normalized amplitude values for these dB levels
        # by inverting our volume_to_db_spl mapping
        db_to_volume = {db: vol for vol, db in self.volume_to_db_spl.items()}

        # Find closest volume values
        initial_volume = min(db_to_volume.items(), key=lambda x: abs(x[0] - initial_db))[1]
        reduced_volume = min(db_to_volume.items(), key=lambda x: abs(x[0] - reduction_db))[1]

        # Calculate volume ratio
        volume_ratio = reduced_volume / initial_volume

        # Calculate sample positions
        reduction_sample = int(reduction_time_minutes * 60 * self.sample_rate)
        fade_samples = int(fade_duration_seconds * self.sample_rate)

        # Create the envelope
        samples = len(audio)
        envelope = np.ones(samples) * initial_volume

        # Check if the reduction point is within the audio duration
        if reduction_sample < samples:
            # Ensure the fade doesn't go beyond the audio duration
            fade_end = min(reduction_sample + fade_samples, samples)
            fade_length = fade_end - reduction_sample

            # Create a smooth fade between initial and reduced volume
            fade_curve = np.linspace(initial_volume, reduced_volume, fade_length)
            envelope[reduction_sample:fade_end] = fade_curve

            # Set the remaining portion to reduced volume
            if fade_end < samples:
                envelope[fade_end:] = reduced_volume

        # Apply the envelope efficiently
        output = apply_modulation(audio, envelope / initial_volume)

        return output

    def apply_auto_shutoff(
        self, audio: np.ndarray, params: SafetyFeatures
    ) -> np.ndarray:
        """
        Apply safety feature to automatically fade out after extended high-volume use.

        Args:
            audio: Input audio array
            params: Parameters for auto shutoff

        Returns:
            Modified audio with shutoff if needed
        """
        if not params or not hasattr(params, "auto_shutoff_minutes"):
            return audio

        # Extract parameters
        shutoff_minutes = params.auto_shutoff_minutes
        threshold_db = params.high_volume_threshold_db

        # Check if our volume exceeds the threshold
        # Look for the closest volume level in our mapping
        db_to_volume = {db: vol for vol, db in self.volume_to_db_spl.items()}
        closest_db = min(db_to_volume.keys(), key=lambda x: abs(x - threshold_db))

        # If we're below threshold, no shutoff needed
        if closest_db < threshold_db:
            return audio

        # Calculate shutoff point in samples
        shutoff_sample = int(shutoff_minutes * 60 * self.sample_rate)

        # Check if shutoff point is within our audio duration
        if shutoff_sample >= len(audio):
            return audio  # No shutoff needed

        # Create fade out envelope
        # 30 second fade-out is reasonable
        fade_samples = min(30 * self.sample_rate, len(audio) - shutoff_sample)

        # Create envelope: ones until shutoff, then fade to zero
        envelope = np.ones(len(audio))
        fade_curve = np.cos(np.linspace(0, np.pi / 2, fade_samples)) ** 2  # Smooth fade
        envelope[shutoff_sample : shutoff_sample + fade_samples] = fade_curve
        envelope[shutoff_sample + fade_samples :] = 0  # Complete silence after fade

        # Apply envelope efficiently
        output = apply_modulation(audio, envelope)

        return output