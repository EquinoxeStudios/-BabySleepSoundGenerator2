"""
Dynamic modulation and variation for audio signals.
"""

import random
import numpy as np
from models.constants import Constants
from utils.optional_imports import HAS_PERLIN, HAS_NUMBA

# Import optional libraries
if HAS_PERLIN:
    import noise

if HAS_NUMBA:
    import numba


class ModulationProcessor:
    """Processor for creating organic modulation and temporal variations."""
    
    def __init__(self, sample_rate: int, use_perlin: bool = True, depth: float = 0.08, rate: float = 0.002):
        """
        Initialize the modulation processor.
        
        Args:
            sample_rate: Audio sample rate
            use_perlin: Whether to use Perlin noise for more natural variations
            depth: Modulation depth (percentage of amplitude variation)
            rate: Modulation rate in Hz (cycles per second)
        """
        self.sample_rate = sample_rate
        self.use_perlin = use_perlin and HAS_PERLIN
        self.modulation_depth = depth
        self.modulation_rate = rate
        
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
        seeds = [random.randint(0, 1000) for _ in range(octaves)]

        # Parameter determines how "organic" the noise feels
        scale_factor = 0.002  # Controls the "speed" of changes

        # Use numba for acceleration if available
        if HAS_NUMBA:
            result = self._generate_perlin_noise_numba(samples, octaves, persistence, scale_factor, seeds)
        else:
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
    
    def _generate_perlin_noise_numba(self, samples, octaves, persistence, scale_factor, seeds):
        """Numba-accelerated Perlin noise generation if available"""
        if not HAS_NUMBA:
            return np.zeros(samples)
            
        @numba.jit(nopython=True)
        def _generate_noise(samples, octaves, persistence, scale_factor, seeds):
            result = np.zeros(samples)
            for i in range(samples):
                value = 0.0
                for j in range(octaves):
                    # Each octave uses a different seed and scale
                    octave_scale = scale_factor * (2**j)
                    # Simplified noise approximation for numba compatibility
                    x = i * octave_scale + seeds[j]
                    n = (np.sin(x) * 43758.5453) % 1
                    value += persistence**j * n
                result[i] = value
            return result
            
        return _generate_noise(samples, octaves, persistence, scale_factor, np.array(seeds))
                
    def apply_dynamic_modulation(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply subtle, organic modulation to prevent listener fatigue.
        This uses Perlin noise to create ultra-slow variations over time.
        
        Args:
            audio: Input audio array
            
        Returns:
            Modulated audio
        """
        samples = len(audio)
        duration_seconds = samples / self.sample_rate

        # Create very slow modulation curve
        if self.use_perlin:
            # Use perlin noise for organic modulation
            mod_curve = self._generate_perlin_noise(duration_seconds, octaves=1, persistence=0.5)
            
            # Stretch the curve to be very slow (only a few cycles over the whole duration)
            indices = np.linspace(0, len(mod_curve) // 100, samples).astype(int)
            mod_curve = mod_curve[indices]
        else:
            # Fallback to sine wave modulation
            t = np.linspace(0, duration_seconds * self.modulation_rate * 2 * np.pi, samples)
            mod_curve = np.sin(t)

        # Scale to desired modulation depth
        mod_curve = 1.0 + self.modulation_depth * mod_curve

        # Apply the modulation efficiently
        if len(audio.shape) > 1:
            # Stereo
            modulated_audio = audio * mod_curve[:, np.newaxis]
        else:
            # Mono
            modulated_audio = audio * mod_curve

        # Normalize if needed
        max_val = np.max(np.abs(modulated_audio))
        if max_val > Constants.MAX_AUDIO_VALUE:
            modulated_audio = modulated_audio / max_val * Constants.MAX_AUDIO_VALUE

        return modulated_audio