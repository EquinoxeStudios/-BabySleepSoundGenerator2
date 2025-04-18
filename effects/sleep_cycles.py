"""
Sleep cycle modulation effects.
"""

import numpy as np
from models.parameters import SleepCycleModulation
from utils.optional_imports import HAS_PERLIN

# Import optional libraries
if HAS_PERLIN:
    import noise


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
            perlin = self._generate_perlin_noise(samples / self.sample_rate, octaves=1, persistence=0.5)

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
        if is_stereo:
            output = audio * modulation[:, np.newaxis]
        else:
            output = audio * modulation

        return output
        
    def _generate_perlin_noise(
        self, duration_seconds: int, octaves: int = 4, persistence: float = 0.5
    ) -> np.ndarray:
        """
        Generate organic noise using Perlin/Simplex noise algorithm.
        This creates more natural textures than basic random noise.

        Args:
            duration_seconds: Length of the audio in seconds
            octaves: Number of layers of detail
            persistence: How much each octave contributes to the overall shape

        Returns:
            Numpy array of noise with natural patterns
        """
        if not HAS_PERLIN:
            # Fall back to regular noise if library not available
            return np.random.normal(0, 0.5, int(duration_seconds * self.sample_rate))

        samples = int(duration_seconds * self.sample_rate)
        result = np.zeros(samples)

        # Create seeds for each octave
        seeds = [np.random.randint(0, 1000) for _ in range(octaves)]

        # Parameter determines how "organic" the noise feels
        scale_factor = 0.002  # Controls the "speed" of changes

        # Standard implementation
        for i in range(samples):
            value = 0
            for j in range(octaves):
                # Each octave uses a different seed and scale
                octave_scale = scale_factor * (2**j)
                value += persistence**j * noise.pnoise1(i * octave_scale, base=seeds[j])

            result[i] = value

        # Normalize to +/- 0.5 range
        result = 0.5 * result / np.max(np.abs(result))

        return result.astype(np.float32)