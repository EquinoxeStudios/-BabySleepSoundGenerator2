"""
Noise generators for various noise colors (white, pink, brown).
"""

import logging
import numpy as np
from scipy import signal

from sound_profiles.base import SoundProfileGenerator
from utils.optional_imports import HAS_PERLIN
# Direct import from perlin_utils instead of through utils.__init__
from utils.perlin_utils import generate_perlin_noise, apply_modulation, generate_dynamic_modulation

logger = logging.getLogger("BabySleepSoundGenerator")


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
        
    def generate(self, duration_seconds: int, sound_type: str = "white", **kwargs) -> np.ndarray:
        """
        Generate specified noise color.
        
        Args:
            duration_seconds: Duration in seconds
            sound_type: Type of noise to generate (white, pink, brown)
            **kwargs: Additional parameters
            
        Returns:
            Noise array
        """
        if sound_type == "white":
            return self.generate_white_noise(duration_seconds, **kwargs)
        elif sound_type == "pink":
            return self.generate_pink_noise_fft(duration_seconds, **kwargs)
        elif sound_type == "brown":
            return self.generate_brown_noise_fft(duration_seconds, **kwargs)
        else:
            raise ValueError(f"Unknown noise type: {sound_type}")
    
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

        # Generate modulation curve
        mod_curve = generate_dynamic_modulation(
            self.sample_rate,
            duration_seconds,
            depth=self.modulation_depth,
            use_perlin=self.use_perlin
        )

        # Apply the modulation efficiently
        modulated_audio = apply_modulation(audio, mod_curve)

        # Normalize if needed
        max_val = np.max(np.abs(modulated_audio))
        if max_val > 0.95:  # Constants.MAX_AUDIO_VALUE
            modulated_audio = modulated_audio / max_val * 0.95

        return modulated_audio

    def generate_white_noise(self, duration_seconds: int, **kwargs) -> np.ndarray:
        """
        Generate enhanced white noise with natural texture.
        Uses Perlin noise if available, otherwise falls back to traditional method.
        
        Args:
            duration_seconds: Length of the sound in seconds
            **kwargs: Additional parameters
            
        Returns:
            White noise array
        """
        samples = int(duration_seconds * self.sample_rate)

        if self.use_perlin and HAS_PERLIN:
            # Use Perlin noise for a more organic texture
            noise_array = generate_perlin_noise(
                self.sample_rate, 
                duration_seconds, 
                octaves=6, 
                persistence=0.7
            )

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

    def generate_pink_noise_fft(self, duration_seconds: int, **kwargs) -> np.ndarray:
        """
        Generate pink noise using FFT-based spectral shaping.
        This creates more accurate 1/f spectrum than filtered white noise.
        
        Args:
            duration_seconds: Length of the sound in seconds
            **kwargs: Additional parameters
            
        Returns:
            Pink noise array
        """
        samples = int(duration_seconds * self.sample_rate)

        # Start with white noise
        if self.use_perlin and HAS_PERLIN:
            white = generate_perlin_noise(
                self.sample_rate, 
                duration_seconds, 
                octaves=6, 
                persistence=0.7
            )
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

    def generate_brown_noise_fft(self, duration_seconds: int, **kwargs) -> np.ndarray:
        """
        Generate brown/red noise using FFT-based spectral shaping.
        Brown noise has energy proportional to 1/f²
        
        Args:
            duration_seconds: Length of the sound in seconds
            **kwargs: Additional parameters
            
        Returns:
            Brown noise array
        """
        samples = int(duration_seconds * self.sample_rate)

        # Start with white noise
        if self.use_perlin and HAS_PERLIN:
            white = generate_perlin_noise(
                self.sample_rate, 
                duration_seconds, 
                octaves=6, 
                persistence=0.7
            )
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