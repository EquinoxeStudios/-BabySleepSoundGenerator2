"""
Breathing modulation effects.
"""

import numpy as np
from models.parameters import BreathingModulation
from utils.optional_imports import HAS_PERLIN

# Import optional libraries
if HAS_PERLIN:
    import noise


class BreathingModulator:
    """Applies breathing rhythm modulation to audio."""
    
    def __init__(self, sample_rate: int, use_perlin: bool = True):
        """
        Initialize the breathing modulator.
        
        Args:
            sample_rate: Audio sample rate
            use_perlin: Whether to use Perlin noise for more natural variations
        """
        self.sample_rate = sample_rate
        self.use_perlin = use_perlin and HAS_PERLIN
        
    def apply_breathing_modulation(
        self, audio: np.ndarray, params: BreathingModulation
    ) -> np.ndarray:
        """
        Apply subtle modulation matching maternal breathing rhythm.

        Args:
            audio: Input audio array
            params: Parameters for breathing modulation

        Returns:
            Modulated audio
        """
        if not params or not params.enabled:
            return audio

        # Extract parameters
        cpm = params.cycles_per_minute

        # Convert to Hz
        breathing_freq = cpm / 60.0

        # Generate the modulation envelope
        samples = len(audio)
        is_stereo = len(audio.shape) > 1

        # Time points
        t = np.arange(samples) / self.sample_rate

        if HAS_PERLIN and self.use_perlin:
            # Generate perlin noise sampled at breathing frequency
            # for more natural, organic variations
            perlin = self._generate_perlin_noise(
                samples / self.sample_rate / 5, octaves=1, persistence=0.5
            )

            # Create breathing cycles with slight organic variation
            n_cycles = int(samples / self.sample_rate * breathing_freq) + 1
            indices = np.linspace(0, len(perlin) - 1, n_cycles * 10)
            indices = np.clip(indices.astype(int), 0, len(perlin) - 1)
            cycle_variation = perlin[indices] * 0.1  # 10% variation

            # Create breathing pattern with base frequency + variation
            breathing = np.sin(2 * np.pi * breathing_freq * t)

            # Apply variations by warping the time dimension slightly
            warped_breathing = np.zeros_like(breathing)
            warp_amount = 0.1  # 10% time warping at most

            # Create warped time vector
            warped_t = t.copy()
            warp_freq = breathing_freq / 5  # Slower variation
            warp = warp_amount * np.sin(2 * np.pi * warp_freq * t)
            warped_t += warp

            # Sample breathing pattern at warped time points
            breathing_cycle = np.sin(2 * np.pi * breathing_freq * warped_t)

            # Scale to subtle modulation range (5% intensity variation)
            modulation = 1.0 + 0.05 * breathing_cycle
        else:
            # Simple sinusoidal modulation if Perlin not available
            modulation = 1.0 + 0.05 * np.sin(2 * np.pi * breathing_freq * t)

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