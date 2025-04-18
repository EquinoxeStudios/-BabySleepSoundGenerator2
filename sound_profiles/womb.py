"""
Womb-related sound generators like womb sounds and umbilical swish.
"""

import random
import numpy as np
from scipy import signal
import logging

from sound_profiles.base import SoundProfileGenerator
from utils.optional_imports import HAS_PERLIN

logger = logging.getLogger("BabySleepSoundGenerator")

# Import optional libraries
if HAS_PERLIN:
    import noise


class WombSoundGenerator(SoundProfileGenerator):
    """Generator for womb-related sounds like womb ambience and umbilical swish."""
    
    def __init__(self, sample_rate: int, use_perlin: bool = True):
        """
        Initialize the womb sound generator.
        
        Args:
            sample_rate: Audio sample rate
            use_perlin: Whether to use Perlin noise for more natural variations
        """
        super().__init__(sample_rate, use_perlin)

    def generate(self, duration_seconds: int, **kwargs) -> np.ndarray:
        """
        Generate a womb-related sound based on the specified type.
        
        Args:
            duration_seconds: Duration in seconds
            **kwargs: Additional parameters including 'sound_type'
            
        Returns:
            Sound profile as numpy array
        """
        sound_type = kwargs.get('sound_type', 'womb')
        
        if sound_type == 'womb':
            return self.generate_womb_simulation(duration_seconds)
        elif sound_type == 'umbilical_swish':
            return self.generate_umbilical_swish(duration_seconds)
        else:
            raise ValueError(f"Unknown womb sound type: {sound_type}")

    def generate_womb_simulation(self, duration_seconds: int) -> np.ndarray:
        """
        Simulate womb sounds with enhanced realism
        
        Args:
            duration_seconds: Length of the sound in seconds
            
        Returns:
            Womb simulation sound array
        """
        samples = int(duration_seconds * self.sample_rate)

        # Generate base brown noise
        # Simple method for brown noise using integration of white noise
        white_noise = np.random.normal(0, 1, samples)
        brown_noise = np.cumsum(white_noise)
        
        # High pass filter to avoid DC build-up
        b, a = signal.butter(1, 20/(self.sample_rate/2), 'high')
        brown_noise = signal.lfilter(b, a, brown_noise)
        
        # Normalize
        brown_noise = brown_noise / np.max(np.abs(brown_noise)) * 0.9
        base_noise = brown_noise

        # Add a slight rhythmic modulation to simulate blood flow and maternal heartbeat
        t = np.linspace(0, duration_seconds, samples, endpoint=False)

        # Create more organic, natural variations for maternal sounds
        if HAS_PERLIN and self.use_perlin:
            # Generate slow perlin noise for breathing rhythm
            breathing_noise = self._generate_perlin_noise(duration_seconds, octaves=1, persistence=0.5)
            
            # Scale to appropriate breathing rate (12-20 breaths per minute)
            breathing_scale = 16 / 60  # breaths per second
            indices = np.linspace(0, len(breathing_noise) // 40, samples)
            indices = np.clip(indices.astype(int), 0, len(breathing_noise) - 1)
            breathing_modulation = 0.15 * breathing_noise[indices]

            # Add a secondary slower modulation for deeper bodily rhythms
            deep_rhythm_noise = self._generate_perlin_noise(duration_seconds, octaves=1, persistence=0.5)
            deep_indices = np.linspace(0, len(deep_rhythm_noise) // 200, samples)
            deep_indices = np.clip(deep_indices.astype(int), 0, len(deep_rhythm_noise) - 1)
            deep_modulation = 0.1 * deep_rhythm_noise[deep_indices]

            # Combine modulations
            combined_modulation = breathing_modulation + deep_modulation
        else:
            # Fallback to sine-based modulation
            # Slow rhythm for maternal breathing (about 12-20 breaths per minute)
            breathing_rate = 16 / 60  # breaths per second
            breathing_modulation = 0.15 * np.sin(2 * np.pi * breathing_rate * t)
            
            # Add a slower modulation for blood flow
            blood_rate = 4 / 60  # cycles per second
            blood_modulation = 0.1 * np.sin(2 * np.pi * blood_rate * t)
            
            # Combine modulations
            combined_modulation = breathing_modulation + blood_modulation

        # Apply a low-pass filter to make it more womb-like
        # Research-backed cutoff at 1000 Hz to mimic womb frequency filtering
        b, a = signal.butter(6, 1000 / (self.sample_rate / 2), "low")
        filtered_noise = signal.lfilter(b, a, base_noise)

        # Combine with the modulations
        womb_sound = filtered_noise * (1 + combined_modulation)

        # Normalize
        max_val = np.max(np.abs(womb_sound))
        if max_val > 0:
            womb_sound = womb_sound / max_val * 0.9

        return womb_sound

    def generate_umbilical_swish(self, duration_seconds: int) -> np.ndarray:
        """
        Generate umbilical blood flow 'whooshing' sounds.
        This is based on research on intrauterine sounds audible to fetuses.
        
        Args:
            duration_seconds: Length of the sound in seconds
            
        Returns:
            Umbilical swish sound array
        """
        samples = int(duration_seconds * self.sample_rate)

        # Start with filtered brown noise as the base
        # Simple method for brown noise
        white_noise = np.random.normal(0, 1, samples)
        brown_noise = np.cumsum(white_noise)
        
        # High pass filter to avoid DC build-up
        b, a = signal.butter(1, 20/(self.sample_rate/2), 'high')
        brown_noise = signal.lfilter(b, a, brown_noise)
        
        # Normalize
        brown_noise = brown_noise / np.max(np.abs(brown_noise)) * 0.9

        # Apply a band-pass filter to focus on 'whoosh' frequencies (50-150 Hz)
        b, a = signal.butter(4, [50 / (self.sample_rate / 2), 150 / (self.sample_rate / 2)], "band")
        whoosh_base = signal.lfilter(b, a, brown_noise)

        # Create pulsing effect synchronized with heartbeat rate (maternal ~70 bpm)
        t = np.linspace(0, duration_seconds, samples, endpoint=False)
        pulse_rate = 70 / 60  # Convert bpm to Hz

        if HAS_PERLIN and self.use_perlin:
            # Use perlin noise for organic, natural pulse variations
            pulse_noise = self._generate_perlin_noise(duration_seconds / 2, octaves=2, persistence=0.5)
            
            # Map to reasonable range and stretch
            indices = np.linspace(0, len(pulse_noise) - 1, samples)
            indices = np.clip(indices.astype(int), 0, len(pulse_noise) - 1)
            pulse_factor = 0.5 + 0.5 * (pulse_noise[indices] * 0.5 + 0.5)  # Map to 0.25-0.75 range
        else:
            # Use modified sine for the pulse
            pulse_factor = 0.5 + 0.25 * np.sin(2 * np.pi * pulse_rate * t)
            
            # Add a small random variation
            small_variation = 0.1 * np.random.randn(len(pulse_factor))
            
            # Smooth the random variations
            small_variation = np.convolve(
                small_variation,
                np.ones(int(0.05 * self.sample_rate)) / int(0.05 * self.sample_rate),
                mode="same",
            )
            pulse_factor += small_variation

        # Apply the pulsing modulation
        whoosh_pulsed = whoosh_base * pulse_factor

        # Add a second layer with slight frequency shift for richness
        # Filter a different frequency band (100-250 Hz)
        b2, a2 = signal.butter(4, [100 / (self.sample_rate / 2), 250 / (self.sample_rate / 2)], "band")
        whoosh_high = signal.lfilter(b2, a2, brown_noise)

        # Create a different pulse pattern, slightly offset
        if HAS_PERLIN and self.use_perlin:
            pulse_noise2 = self._generate_perlin_noise(duration_seconds / 2, octaves=2, persistence=0.6)
            indices2 = np.linspace(0, len(pulse_noise2) - 1, samples)
            indices2 = np.clip(indices2.astype(int), 0, len(pulse_noise2) - 1)
            pulse_factor2 = 0.4 + 0.3 * (pulse_noise2[indices2] * 0.5 + 0.5)  # Different range
        else:
            # Slight phase shift for second layer
            pulse_factor2 = 0.4 + 0.3 * np.sin(2 * np.pi * pulse_rate * t + 0.5)
            small_variation2 = 0.1 * np.random.randn(len(pulse_factor2))
            small_variation2 = np.convolve(
                small_variation2,
                np.ones(int(0.05 * self.sample_rate)) / int(0.05 * self.sample_rate),
                mode="same",
            )
            pulse_factor2 += small_variation2

        whoosh_pulsed2 = whoosh_high * pulse_factor2

        # Mix both layers
        umbilical_sound = 0.7 * whoosh_pulsed + 0.3 * whoosh_pulsed2

        # Add subtle low-frequency pressure variations
        if HAS_PERLIN and self.use_perlin:
            pressure_variation = self._generate_perlin_noise(duration_seconds / 10, octaves=1, persistence=0.5)
            indices3 = np.linspace(0, len(pressure_variation) - 1, samples)
            indices3 = np.clip(indices3.astype(int), 0, len(pressure_variation) - 1)
            pressure_factor = 0.9 + 0.1 * pressure_variation[indices3]
        else:
            # Very slow modulation for pressure changes (0.05 Hz - complete cycle every 20 seconds)
            pressure_factor = 0.9 + 0.1 * np.sin(2 * np.pi * 0.05 * t)

        umbilical_sound = umbilical_sound * pressure_factor

        # Apply final band-shaping filter
        b_final, a_final = signal.butter(2, 300 / (self.sample_rate / 2), "low")
        umbilical_sound = signal.lfilter(b_final, a_final, umbilical_sound)

        # Normalize
        max_val = np.max(np.abs(umbilical_sound))
        if max_val > 0:
            umbilical_sound = umbilical_sound / max_val * 0.9

        return umbilical_sound
        
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