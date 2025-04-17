"""
Noise generators for various noise colors (white, pink, brown).
"""

import logging
import random
import numpy as np
from scipy import signal

from sound_profiles.base import SoundProfileGenerator
from utils.optional_imports import HAS_PERLIN, HAS_NUMBA

logger = logging.getLogger("BabySleepSoundGenerator")

# Import optional libraries
if HAS_PERLIN:
    import noise
    
if HAS_NUMBA:
    import numba


class NoiseGenerator(SoundProfileGenerator):
    """Generator for different colors of noise."""
    
    def __init__(self, sample_rate: int, use_perlin: bool = True, modulation_depth: float = 0.08):
        """
        Initialize the noise generator.
        
        Args:
            sample_rate: Audio sample rate
            use_perlin: Whether to use Perlin noise for more natural variations
            modulation_depth: Depth of modulation for dynamic variation
        """
        super().__init__(sample_rate, use_perlin)
        self.modulation_depth = modulation_depth
        
    def generate(self, duration_seconds: int, noise_type: str = "white", **kwargs) -> np.ndarray:
        """
        Generate specified noise color.
        
        Args:
            duration_seconds: Duration in seconds
            noise_type: Type of noise to generate (white, pink, brown)
            **kwargs: Additional parameters
            
        Returns:
            Noise array
        """
        if noise_type == "white":
            return self.generate_white_noise(duration_seconds)
        elif noise_type == "pink":
            return self.generate_pink_noise_fft(duration_seconds)
        elif noise_type == "brown":
            return self.generate_brown_noise_fft(duration_seconds)
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
    
    def _apply_dynamic_modulation(self, audio: np.ndarray) -> np.ndarray:
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
        if HAS_PERLIN and self.use_perlin:
            # Use perlin noise for organic modulation
            mod_curve = self._generate_perlin_noise(duration_seconds, octaves=1, persistence=0.5)
            
            # Stretch the curve to be very slow (only a few cycles over the whole duration)
            indices = np.linspace(0, len(mod_curve) // 100, samples).astype(int)
            mod_curve = mod_curve[indices]
        else:
            # Fallback to sine wave modulation
            t = np.linspace(0, duration_seconds * 0.002 * 2 * np.pi, samples)
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
        if max_val > 0.95:  # Constants.MAX_AUDIO_VALUE
            modulated_audio = modulated_audio / max_val * 0.95

        return modulated_audio

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

    def generate_white_noise(self, duration_seconds: int) -> np.ndarray:
        """
        Generate enhanced white noise with natural texture.
        Uses Perlin noise if available, otherwise falls back to traditional method.
        
        Args:
            duration_seconds: Length of the sound in seconds
            
        Returns:
            White noise array
        """
        samples = int(duration_seconds * self.sample_rate)

        if self.use_perlin and HAS_PERLIN:
            # Use Perlin noise for a more organic texture
            noise_array = self._generate_perlin_noise(duration_seconds, octaves=6, persistence=0.7)

            # Apply whitening filter to ensure flat spectrum using FFT
            spectrum = np.fft.rfft(noise_array)
            magnitude = np.abs(spectrum)
            phase = np.angle(spectrum)
            flat_spectrum = np.ones_like(magnitude) * np.exp(1j * phase)
            noise_array = np.fft.irfft(flat_spectrum, n=len(noise_array))

            # Normalize
            noise_array = 0.5 * noise_array / np.max(np.abs(noise_array))
        else:
            # Traditional white noise generation
            noise_array = np.random.normal(0, 0.5, samples)

        # Apply subtle dynamic modulation to prevent fatigue
        noise_array = self._apply_dynamic_modulation(noise_array)

        return noise_array

    def generate_pink_noise_fft(self, duration_seconds: int) -> np.ndarray:
        """
        Generate pink noise using FFT-based spectral shaping.
        This creates more accurate 1/f spectrum than filtered white noise.
        
        Args:
            duration_seconds: Length of the sound in seconds
            
        Returns:
            Pink noise array
        """
        samples = int(duration_seconds * self.sample_rate)

        # Start with white noise
        if self.use_perlin and HAS_PERLIN:
            white = self._generate_perlin_noise(duration_seconds, octaves=6, persistence=0.7)
        else:
            white = np.random.normal(0, 0.5, samples)

        # FFT to frequency domain
        X = np.fft.rfft(white)

        # Generate frequency array
        freqs = np.fft.rfftfreq(samples, 1 / self.sample_rate)

        # Create 1/f filter (pink noise has energy proportional to 1/f)
        # Add small constant to avoid division by zero
        pink_filter = 1 / np.sqrt(freqs + 1e-6)

        # Apply filter while preserving phase
        magnitude = np.abs(X)
        phase = np.angle(X)
        X_pink = pink_filter * magnitude * np.exp(1j * phase)

        # Back to time domain
        pink_noise = np.fft.irfft(X_pink, n=samples)

        # Normalize
        pink_noise = 0.5 * pink_noise / np.max(np.abs(pink_noise))

        # Apply subtle dynamic modulation to prevent fatigue
        pink_noise = self._apply_dynamic_modulation(pink_noise)

        return pink_noise

    def generate_brown_noise_fft(self, duration_seconds: int) -> np.ndarray:
        """
        Generate brown/red noise using FFT-based spectral shaping.
        Brown noise has energy proportional to 1/f²
        
        Args:
            duration_seconds: Length of the sound in seconds
            
        Returns:
            Brown noise array
        """
        samples = int(duration_seconds * self.sample_rate)

        # Start with white noise
        if self.use_perlin and HAS_PERLIN:
            white = self._generate_perlin_noise(duration_seconds, octaves=6, persistence=0.7)
        else:
            white = np.random.normal(0, 0.5, samples)

        # FFT to frequency domain
        X = np.fft.rfft(white)

        # Generate frequency array
        freqs = np.fft.rfftfreq(samples, 1 / self.sample_rate)

        # Create 1/f² filter (brown noise has energy proportional to 1/f²)
        # Add small constant to avoid division by zero
        brown_filter = 1 / (freqs + 1e-6)

        # Apply filter while preserving phase
        magnitude = np.abs(X)
        phase = np.angle(X)
        X_brown = brown_filter * magnitude * np.exp(1j * phase)

        # Back to time domain
        brown_noise = np.fft.irfft(X_brown, n=samples)

        # Normalize
        brown_noise = 0.5 * brown_noise / np.max(np.abs(brown_noise))

        # Apply subtle dynamic modulation
        brown_noise = self._apply_dynamic_modulation(brown_noise)

        return brown_noise