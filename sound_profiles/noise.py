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
from utils.random_state import RandomStateManager
from models.constants import PerformanceConstants

logger = logging.getLogger("BabySleepSoundGenerator")


class NoiseGenerator(SoundProfileGenerator):
    """Generator for different colors of noise."""
    
    def __init__(self, sample_rate: int, use_perlin: bool = True, modulation_depth: float = 0.08, seed: int = None):
        """
        Initialize the noise generator.
        
        Args:
            sample_rate: Audio sample rate
            use_perlin: Whether to use Perlin noise for more natural variations
            modulation_depth: Depth of modulation for dynamic variation
            seed: Random seed for reproducibility
        """
        super().__init__(sample_rate, use_perlin)
        self.modulation_depth = modulation_depth
        self.random_state = RandomStateManager.get_instance(seed)
        
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
            use_perlin=self.use_perlin,
            seed=self.random_state.seed
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

        # For very long durations, process in chunks
        if samples > PerformanceConstants.MAX_DURATION_SECONDS_BEFORE_CHUNKING * self.sample_rate:
            return self._generate_white_noise_chunked(duration_seconds, **kwargs)

        if self.use_perlin and HAS_PERLIN:
            # Use Perlin noise for a more organic texture
            noise_array = generate_perlin_noise(
                self.sample_rate, 
                duration_seconds, 
                octaves=6, 
                persistence=0.7,
                seed=self.random_state.seed
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
            noise_array = self.random_state.normal(0, 0.5, samples)

        # Apply subtle dynamic modulation to prevent fatigue
        noise_array = self._apply_dynamic_modulation(noise_array)

        return noise_array

    def _generate_white_noise_chunked(self, duration_seconds: int, **kwargs) -> np.ndarray:
        """Generate white noise in chunks for very long durations."""
        # Calculate total samples
        total_samples = int(duration_seconds * self.sample_rate)
        
        # Process in chunks
        chunk_seconds = PerformanceConstants.FFT_CHUNK_SIZE_SECONDS
        chunk_samples = int(chunk_seconds * self.sample_rate)
        crossfade_samples = int(PerformanceConstants.CROSSFADE_BETWEEN_CHUNKS_SECONDS * self.sample_rate)
        
        # Calculate number of chunks
        num_chunks = (total_samples + chunk_samples - 1) // chunk_samples
        
        # Create result array
        result = np.zeros(total_samples)
        
        logger.info(f"Generating white noise in {num_chunks} chunks of {chunk_seconds}s each")
        
        # Process each chunk
        for i in range(num_chunks):
            # Calculate chunk boundaries with overlap for crossfade
            start_idx = i * chunk_samples
            end_idx = min(start_idx + chunk_samples + crossfade_samples, total_samples)
            chunk_duration = (end_idx - start_idx) / self.sample_rate
            
            # Generate this chunk
            chunk = self.generate_white_noise(chunk_duration)
            
            # Apply crossfade if not the first chunk
            if i > 0:
                # Overlap with previous chunk
                overlap_start = start_idx
                overlap_end = min(start_idx + crossfade_samples, total_samples)
                
                # Create crossfade weights
                fade_in = np.linspace(0, 1, overlap_end - overlap_start)
                fade_out = 1 - fade_in
                
                # Apply crossfade
                result[overlap_start:overlap_end] = (
                    result[overlap_start:overlap_end] * fade_out + 
                    chunk[:overlap_end - overlap_start] * fade_in
                )
                
                # Add the non-overlapping part
                if overlap_end < end_idx:
                    result[overlap_end:end_idx] = chunk[overlap_end - overlap_start:end_idx - overlap_start]
            else:
                # First chunk, no crossfade needed
                result[start_idx:end_idx] = chunk[:end_idx - start_idx]
        
        return result

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
        
        # For very long durations, process in chunks
        if samples > PerformanceConstants.MAX_DURATION_SECONDS_BEFORE_CHUNKING * self.sample_rate:
            return self._generate_noise_fft_chunked(duration_seconds, noise_type="pink", **kwargs)

        # Start with white noise
        if self.use_perlin and HAS_PERLIN:
            white = generate_perlin_noise(
                self.sample_rate, 
                duration_seconds, 
                octaves=6, 
                persistence=0.7,
                seed=self.random_state.seed
            )
        else:
            white = self.random_state.normal(0, 0.5, samples)

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
        
        # For very long durations, process in chunks
        if samples > PerformanceConstants.MAX_DURATION_SECONDS_BEFORE_CHUNKING * self.sample_rate:
            return self._generate_noise_fft_chunked(duration_seconds, noise_type="brown", **kwargs)

        # Start with white noise
        if self.use_perlin and HAS_PERLIN:
            white = generate_perlin_noise(
                self.sample_rate, 
                duration_seconds, 
                octaves=6, 
                persistence=0.7,
                seed=self.random_state.seed
            )
        else:
            white = self.random_state.normal(0, 0.5, samples)

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
        
    def _generate_noise_fft_chunked(self, duration_seconds: int, noise_type: str, **kwargs) -> np.ndarray:
        """Generate colored noise in chunks for very long durations."""
        # Calculate total samples
        total_samples = int(duration_seconds * self.sample_rate)
        
        # Process in chunks
        chunk_seconds = PerformanceConstants.FFT_CHUNK_SIZE_SECONDS
        chunk_samples = int(chunk_seconds * self.sample_rate)
        crossfade_samples = int(PerformanceConstants.CROSSFADE_BETWEEN_CHUNKS_SECONDS * self.sample_rate)
        
        # Calculate number of chunks
        num_chunks = (total_samples + chunk_samples - 1) // chunk_samples
        
        # Create result array
        result = np.zeros(total_samples)
        
        logger.info(f"Generating {noise_type} noise in {num_chunks} chunks of {chunk_seconds}s each")
        
        # Process each chunk
        for i in range(num_chunks):
            # Calculate chunk boundaries with overlap for crossfade
            start_idx = i * chunk_samples
            end_idx = min(start_idx + chunk_samples + crossfade_samples, total_samples)
            chunk_duration = (end_idx - start_idx) / self.sample_rate
            
            # Generate this chunk based on noise type
            if noise_type == "pink":
                chunk = self.generate_pink_noise_fft(chunk_duration)
            elif noise_type == "brown":
                chunk = self.generate_brown_noise_fft(chunk_duration)
            else:
                chunk = self.generate_white_noise(chunk_duration)
            
            # Apply crossfade if not the first chunk
            if i > 0:
                # Overlap with previous chunk
                overlap_start = start_idx
                overlap_end = min(start_idx + crossfade_samples, total_samples)
                
                # Create crossfade weights
                fade_in = np.linspace(0, 1, overlap_end - overlap_start)
                fade_out = 1 - fade_in
                
                # Apply crossfade
                result[overlap_start:overlap_end] = (
                    result[overlap_start:overlap_end] * fade_out + 
                    chunk[:overlap_end - overlap_start] * fade_in
                )
                
                # Add the non-overlapping part
                if overlap_end < end_idx:
                    result[overlap_end:end_idx] = chunk[overlap_end - overlap_start:end_idx - overlap_start]
            else:
                # First chunk, no crossfade needed
                result[start_idx:end_idx] = chunk[:end_idx - start_idx]
        
        return result