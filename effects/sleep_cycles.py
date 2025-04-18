"""
Sleep cycle modulation effects.
"""

import numpy as np
from models.parameters import SleepCycleModulation
from utils.optional_imports import HAS_PERLIN
# Direct import from perlin_utils instead of through utils.__init__
from utils.perlin_utils import generate_perlin_noise, apply_modulation


class SleepCycleModulator:
    """Applies sleep cycle modulation to audio."""
    
    def __init__(self, sample_rate: int, use_perlin: bool = True):
        """
        Initialize the sleep cycle modulator.
        
        Args:
            sample_rate: Audio sample rate
            use_perlin: Whether to use Perlin noise for more natural variations
        """
        self.sample_rate = sample_rate
        self.use_perlin = use_perlin and HAS_PERLIN
        
    def apply_sleep_cycle_modulation(
        self, audio: np.ndarray, params: SleepCycleModulation
    ) -> np.ndarray:
        """
        Apply cyclical intensity modulation to align with infant sleep cycles.

        Args:
            audio: Input audio array
            params: Parameters for sleep cycle modulation

        Returns:
            Modulated audio
        """
        if not params or not params.enabled:
            return audio

        # Extract parameters
        cycle_minutes = params.cycle_minutes

        # Create modulation signal
        samples = len(audio)
        is_stereo = len(audio.shape) > 1

        # Convert cycle duration to samples
        cycle_samples = int(cycle_minutes * 60 * self.sample_rate)

        # Create the modulation array using sine wave or perlin noise
        if HAS_PERLIN and self.use_perlin:
            # Generate perlin noise for more natural variation
            duration_seconds = samples / self.sample_rate
            perlin = generate_perlin_noise(
                self.sample_rate, 
                duration_seconds, 
                octaves=1, 
                persistence=0.5
            )

            # Stretch to desired cycle length
            cycle_points = int(samples / cycle_samples) + 1
            indices = np.linspace(0, len(perlin) - 1, cycle_points)
            indices = np.clip(indices.astype(int), 0, len(perlin) - 1)
            cycle_curve = perlin[indices]

            # Interpolate to full length
            x_points = np.linspace(0, cycle_points, len(cycle_curve))
            x_interp = np.linspace(0, cycle_points, samples)
            modulation = np.interp(x_interp, x_points, cycle_curve)

            # Scale to desired range (0.85 to 1.15) - subtle 15% modulation
            modulation = 1.0 + 0.15 * modulation
        else:
            # Use smooth sine wave as fallback
            cycle_freq = 1 / (cycle_minutes * 60)  # Hz
            t = np.arange(samples) / self.sample_rate
            modulation = 1.0 + 0.1 * np.sin(2 * np.pi * cycle_freq * t)  # 10% modulation

        # Apply modulation efficiently
        output = apply_modulation(audio, modulation)

        return output