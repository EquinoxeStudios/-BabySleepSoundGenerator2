"""
Womb-related sound generators like womb sounds and umbilical swish.
"""

import numpy as np
from scipy import signal
import logging
from typing import Optional, Dict, Any, Union

from sound_profiles.base import SoundProfileGenerator
from utils.optional_imports import HAS_PERLIN
# Import from utils instead of direct import
from utils.perlin_utils import generate_perlin_noise, apply_modulation
from utils.random_state import RandomStateManager
from models.constants import WombConstants, UmbilicalConstants

logger = logging.getLogger("BabySleepSoundGenerator")


class WombSoundGenerator(SoundProfileGenerator):
    """Generator for womb-related sounds like womb ambience and umbilical swish."""
    
    def __init__(self, sample_rate: int, use_perlin: bool = True, seed: Optional[int] = None, **kwargs):
        """
        Initialize the womb sound generator.
        
        Args:
            sample_rate: Audio sample rate
            use_perlin: Whether to use Perlin noise for more natural variations
            seed: Random seed for reproducibility
            **kwargs: Additional parameters
        """
        super().__init__(sample_rate, use_perlin, seed, **kwargs)
        self.random_state = RandomStateManager.get_instance(seed)

    def generate(self, duration_seconds: int, **kwargs) -> np.ndarray:
        """
        Generate a womb-related sound based on the specified type.
        
        Args:
            duration_seconds: Duration in seconds
            **kwargs: Additional parameters including sound_type
            
        Returns:
            Sound profile as numpy array
        """
        # Get sound type with default to "womb"
        sound_type = kwargs.get('sound_type', 'womb')
        
        try:
            if sound_type == 'womb':
                return self.generate_womb_simulation(duration_seconds)
            elif sound_type == 'umbilical_swish':
                return self.generate_umbilical_swish(duration_seconds)
            else:
                raise ValueError(f"Unknown womb sound type: {sound_type}")
        except Exception as e:
            logger.error(f"Error generating {sound_type} sound: {e}")
            # Generate simple brown noise as fallback
            return self._generate_fallback_brown_noise(duration_seconds)
            
    def _generate_fallback_brown_noise(self, duration_seconds: int) -> np.ndarray:
        """Generate simple brown noise as fallback in case of error."""
        try:
            samples = int(duration_seconds * self.sample_rate)
            
            # Simple brown noise using a safer method to avoid overflow
            white_noise = self.random_state.normal(0, 1, samples)
            
            # Use a running sum with decay to prevent unbounded growth
            brown_noise = np.zeros(samples)
            decay = 0.99  # Decay factor to prevent overflow
            
            # Running sum with decay
            for i in range(1, samples):
                brown_noise[i] = decay * brown_noise[i-1] + white_noise[i]
            
            # High pass filter to avoid DC build-up
            b, a = signal.butter(1, 20/(self.sample_rate/2), 'high')
            brown_noise = signal.lfilter(b, a, brown_noise)
            
            # Normalize
            max_val = np.max(np.abs(brown_noise))
            if max_val > 0:
                brown_noise = brown_noise / max_val * 0.7
                
            return self.sanitize_audio(brown_noise, max_amplitude=0.7)
        except Exception as e:
            logger.error(f"Error generating fallback brown noise: {e}")
            # Return simple noise as last resort
            return self.random_state.normal(0, 0.3, int(duration_seconds * self.sample_rate))

    def generate_womb_simulation(self, duration_seconds: int) -> np.ndarray:
        """
        Simulate womb sounds with enhanced realism
        
        Args:
            duration_seconds: Length of the sound in seconds
            
        Returns:
            Womb simulation sound array
        """
        try:
            samples = int(duration_seconds * self.sample_rate)

            # Generate base brown noise using safer method
            white_noise = self.random_state.normal(0, 1, samples)
            
            # Use a running sum with decay to prevent unbounded growth
            brown_noise = np.zeros(samples)
            decay = 0.99  # Decay factor to prevent overflow
            
            # Running sum with decay
            for i in range(1, samples):
                brown_noise[i] = decay * brown_noise[i-1] + white_noise[i]
            
            # High pass filter to avoid DC build-up
            b, a = signal.butter(1, 20/(self.sample_rate/2), 'high')
            brown_noise = signal.lfilter(b, a, brown_noise)
            
            # Normalize with safety check
            max_val = np.max(np.abs(brown_noise))
            if max_val > 0:
                brown_noise = brown_noise / max_val * 0.9
            
            base_noise = brown_noise

            # Add a slight rhythmic modulation to simulate blood flow and maternal heartbeat
            t = np.linspace(0, duration_seconds, samples, endpoint=False)

            # Create more organic, natural variations for maternal sounds
            if HAS_PERLIN and self.use_perlin:
                try:
                    # Generate slow perlin noise for breathing rhythm
                    breathing_noise = generate_perlin_noise(
                        self.sample_rate, 
                        duration_seconds, 
                        octaves=1, 
                        persistence=0.5,
                        seed=self.random_state.seed
                    )
                    
                    # Scale to appropriate breathing rate (12-20 breaths per minute)
                    breathing_scale = WombConstants.BREATHING_RATE_BREATHS_PER_MIN / 60  # breaths per second
                    indices = np.linspace(0, len(breathing_noise) // 40, samples)
                    indices = np.clip(indices.astype(int), 0, len(breathing_noise) - 1)
                    breathing_modulation = WombConstants.BREATHING_MODULATION_DEPTH * breathing_noise[indices]

                    # Add a secondary slower modulation for deeper bodily rhythms
                    deep_rhythm_noise = generate_perlin_noise(
                        self.sample_rate, 
                        duration_seconds, 
                        octaves=1, 
                        persistence=0.5,
                        seed=self.random_state.seed + 1 if self.random_state.seed else None
                    )
                    deep_indices = np.linspace(0, len(deep_rhythm_noise) // 200, samples)
                    deep_indices = np.clip(deep_indices.astype(int), 0, len(deep_rhythm_noise) - 1)
                    deep_modulation = WombConstants.DEEP_RHYTHM_MODULATION_DEPTH * deep_rhythm_noise[deep_indices]

                    # Combine modulations and clip to reasonable range
                    combined_modulation = np.clip(breathing_modulation + deep_modulation, -0.3, 0.3)
                except Exception as e:
                    logger.warning(f"Error creating perlin modulation for womb sound: {e}")
                    # Fall back to sine-based modulation
                    breathing_rate = WombConstants.BREATHING_RATE_BREATHS_PER_MIN / 60  # breaths per second
                    breathing_modulation = WombConstants.BREATHING_MODULATION_DEPTH * np.sin(2 * np.pi * breathing_rate * t)
                    
                    blood_rate = WombConstants.BLOOD_FLOW_CYCLES_PER_MIN / 60  # cycles per second
                    blood_modulation = WombConstants.DEEP_RHYTHM_MODULATION_DEPTH * np.sin(2 * np.pi * blood_rate * t)
                    
                    combined_modulation = np.clip(breathing_modulation + blood_modulation, -0.3, 0.3)
            else:
                # Fallback to sine-based modulation
                # Slow rhythm for maternal breathing (about 12-20 breaths per minute)
                breathing_rate = WombConstants.BREATHING_RATE_BREATHS_PER_MIN / 60  # breaths per second
                breathing_modulation = WombConstants.BREATHING_MODULATION_DEPTH * np.sin(2 * np.pi * breathing_rate * t)
                
                # Add a slower modulation for blood flow
                blood_rate = WombConstants.BLOOD_FLOW_CYCLES_PER_MIN / 60  # cycles per second
                blood_modulation = WombConstants.DEEP_RHYTHM_MODULATION_DEPTH * np.sin(2 * np.pi * blood_rate * t)
                
                # Combine modulations and clip to reasonable range
                combined_modulation = np.clip(breathing_modulation + blood_modulation, -0.3, 0.3)

            # Apply a low-pass filter to make it more womb-like
            # Research-backed cutoff at 1000 Hz to mimic womb frequency filtering
            b, a = signal.butter(6, WombConstants.WOMB_LOWPASS_CUTOFF_HZ / (self.sample_rate / 2), "low")
            filtered_noise = signal.lfilter(b, a, base_noise)

            # Combine with the modulations
            womb_sound = filtered_noise * (1 + combined_modulation)

            # Check for any NaN or Inf values
            if np.isnan(womb_sound).any() or np.isinf(womb_sound).any():
                logger.warning("NaN or Inf values detected in womb sound, cleaning up")
                womb_sound = np.nan_to_num(womb_sound, nan=0.0, posinf=0.95, neginf=-0.95)

            # Normalize and sanitize
            return self.sanitize_audio(womb_sound, max_amplitude=WombConstants.DEFAULT_AMPLITUDE)
            
        except Exception as e:
            logger.error(f"Error generating womb simulation: {e}")
            return self._generate_fallback_brown_noise(duration_seconds)

    def generate_umbilical_swish(self, duration_seconds: int) -> np.ndarray:
        """
        Generate umbilical blood flow 'whooshing' sounds.
        This is based on research on intrauterine sounds audible to fetuses.
        
        Args:
            duration_seconds: Length of the sound in seconds
            
        Returns:
            Umbilical swish sound array
        """
        try:
            samples = int(duration_seconds * self.sample_rate)

            # Start with filtered brown noise as the base
            # Simple method for brown noise that avoids overflow
            white_noise = self.random_state.normal(0, 1, samples)
            
            # Safer method to generate brown noise to avoid overflow
            # Use a running sum with decay to prevent unbounded growth
            brown_noise = np.zeros(samples)
            decay = 0.99  # Decay factor to prevent overflow
            
            # Running sum with decay
            for i in range(1, samples):
                brown_noise[i] = decay * brown_noise[i-1] + white_noise[i]
            
            # Apply high pass filter to remove DC offset
            b, a = signal.butter(1, 20/(self.sample_rate/2), 'high')
            brown_noise = signal.lfilter(b, a, brown_noise)
            
            # Normalize to prevent extreme values
            max_val = np.max(np.abs(brown_noise))
            if max_val > 0:
                brown_noise = brown_noise / max_val * 0.9
            
            # Apply a band-pass filter to focus on 'whoosh' frequencies
            b, a = signal.butter(
                4, 
                [UmbilicalConstants.WHOOSH_BANDPASS_LOW_HZ / (self.sample_rate / 2), 
                 UmbilicalConstants.WHOOSH_BANDPASS_HIGH_HZ / (self.sample_rate / 2)], 
                "band"
            )
            whoosh_base = signal.lfilter(b, a, brown_noise)

            # Create pulsing effect synchronized with heartbeat rate (maternal ~70 bpm)
            t = np.linspace(0, duration_seconds, samples, endpoint=False)
            pulse_rate = UmbilicalConstants.PULSE_RATE_BPM / 60  # Convert bpm to Hz

            if HAS_PERLIN and self.use_perlin:
                # Use perlin noise for organic, natural pulse variations
                try:
                    pulse_noise = generate_perlin_noise(
                        self.sample_rate, 
                        duration_seconds / 2, 
                        octaves=2, 
                        persistence=0.5,
                        seed=self.random_state.seed
                    )
                    
                    # Map to reasonable range and stretch
                    indices = np.linspace(0, len(pulse_noise) - 1, samples)
                    indices = np.clip(indices.astype(int), 0, len(pulse_noise) - 1)
                    base = UmbilicalConstants.PULSE_FACTOR_BASE
                    variation = UmbilicalConstants.PULSE_FACTOR_VARIATION
                    pulse_factor = base + variation * np.clip(0.5 + 0.5 * pulse_noise[indices], 0.1, 0.9)
                except Exception as e:
                    logger.warning(f"Error creating perlin modulation for umbilical swish: {e}")
                    # Fall back to sine-based pulse
                    base = UmbilicalConstants.PULSE_FACTOR_BASE
                    variation = UmbilicalConstants.PULSE_FACTOR_VARIATION
                    pulse_factor = base + variation * np.sin(2 * np.pi * pulse_rate * t)
            else:
                # Use modified sine for the pulse
                base = UmbilicalConstants.PULSE_FACTOR_BASE
                variation = UmbilicalConstants.PULSE_FACTOR_VARIATION
                pulse_factor = base + variation * np.sin(2 * np.pi * pulse_rate * t)
                
                # Add a small random variation
                small_variation = 0.1 * self.random_state.normal(0, 1, len(pulse_factor))
                # Smooth the variations
                smooth_window = int(0.05 * self.sample_rate)
                if smooth_window > 0:
                    smoothing_kernel = np.ones(smooth_window) / smooth_window
                    small_variation = np.convolve(small_variation, smoothing_kernel, mode="same")
                pulse_factor += small_variation
            
            # Clip pulse factor to avoid extreme values
            pulse_factor = np.clip(pulse_factor, 0.1, 0.9)

            # Apply the pulsing modulation
            whoosh_pulsed = whoosh_base * pulse_factor

            # Add a second layer with slight frequency shift for richness
            # Filter a different frequency band
            b2, a2 = signal.butter(
                4, 
                [UmbilicalConstants.SECONDARY_BANDPASS_LOW_HZ / (self.sample_rate / 2), 
                 UmbilicalConstants.SECONDARY_BANDPASS_HIGH_HZ / (self.sample_rate / 2)], 
                "band"
            )
            whoosh_high = signal.lfilter(b2, a2, brown_noise)

            # Create a different pulse pattern, slightly offset
            if HAS_PERLIN and self.use_perlin:
                try:
                    pulse_noise2 = generate_perlin_noise(
                        self.sample_rate, 
                        duration_seconds / 2, 
                        octaves=2, 
                        persistence=0.6,
                        seed=self.random_state.seed + 1 if self.random_state.seed else None
                    )
                    indices2 = np.linspace(0, len(pulse_noise2) - 1, samples)
                    indices2 = np.clip(indices2.astype(int), 0, len(pulse_noise2) - 1)
                    pulse_factor2 = 0.4 + 0.3 * np.clip(0.5 + 0.5 * pulse_noise2[indices2], 0.1, 0.7)
                except Exception as e:
                    logger.warning(f"Error creating second perlin modulation for umbilical swish: {e}")
                    # Fall back to sine
                    pulse_factor2 = 0.4 + 0.3 * np.sin(2 * np.pi * pulse_rate * t + 0.5)
            else:
                # Slight phase shift for second layer
                pulse_factor2 = 0.4 + 0.3 * np.sin(2 * np.pi * pulse_rate * t + 0.5)
                small_variation2 = 0.1 * self.random_state.normal(0, 1, len(pulse_factor2))
                # Smooth the variations
                smooth_window = int(0.05 * self.sample_rate)
                if smooth_window > 0:
                    smoothing_kernel = np.ones(smooth_window) / smooth_window
                    small_variation2 = np.convolve(small_variation2, smoothing_kernel, mode="same")
                pulse_factor2 += small_variation2
            
            # Clip to avoid extreme values
            pulse_factor2 = np.clip(pulse_factor2, 0.1, 0.7)

            whoosh_pulsed2 = whoosh_high * pulse_factor2

            # Mix both layers
            umbilical_sound = 0.7 * whoosh_pulsed + 0.3 * whoosh_pulsed2

            # Check for any extreme values
            if np.isnan(umbilical_sound).any() or np.isinf(umbilical_sound).any():
                logger.warning("Found NaN or Inf in umbilical sound, replacing with zeros")
                umbilical_sound = np.nan_to_num(umbilical_sound, nan=0.0, posinf=0.9, neginf=-0.9)
                
            # Add subtle low-frequency pressure variations
            if HAS_PERLIN and self.use_perlin:
                try:
                    pressure_variation = generate_perlin_noise(
                        self.sample_rate, 
                        duration_seconds / 10, 
                        octaves=1, 
                        persistence=0.5,
                        seed=self.random_state.seed + 2 if self.random_state.seed else None
                    )
                    indices3 = np.linspace(0, len(pressure_variation) - 1, samples)
                    indices3 = np.clip(indices3.astype(int), 0, len(pressure_variation) - 1)
                    pressure_factor = 0.9 + 0.1 * pressure_variation[indices3]
                except Exception as e:
                    logger.warning(f"Error creating pressure variation for umbilical swish: {e}")
                    # Fall back to sine
                    pressure_factor = 0.9 + 0.1 * np.sin(2 * np.pi * 0.05 * t)
            else:
                # Very slow modulation for pressure changes (0.05 Hz - complete cycle every 20 seconds)
                pressure_factor = 0.9 + 0.1 * np.sin(2 * np.pi * 0.05 * t)
                
            # Clip to avoid extreme values
            pressure_factor = np.clip(pressure_factor, 0.7, 1.1)

            umbilical_sound = umbilical_sound * pressure_factor

            # Apply final band-shaping filter
            b_final, a_final = signal.butter(2, 300 / (self.sample_rate / 2), "low")
            umbilical_sound = signal.lfilter(b_final, a_final, umbilical_sound)

            # Final sanitization
            return self.sanitize_audio(umbilical_sound, max_amplitude=UmbilicalConstants.DEFAULT_AMPLITUDE)
        
        except Exception as e:
            logger.error(f"Error generating umbilical swish: {e}")
            # Generate a simpler swooshing sound as fallback
            return self._generate_fallback_brown_noise(duration_seconds)