"""
Natural sound generators like heartbeat, shushing, and fan sounds.
"""

import numpy as np
from scipy import signal
import logging
from typing import Optional, Dict, Any, Union, Callable, Tuple

from sound_profiles.base import SoundProfileGenerator
from models.parameters import HeartbeatParameters, DynamicShushing, ParentalVoice
from utils.optional_imports import HAS_PERLIN
# Import from utils instead of direct import
from utils.perlin_utils import generate_perlin_noise, apply_modulation
from utils.random_state import RandomStateManager
from models.constants import (
    HeartbeatConstants, ShushingConstants, FanConstants, 
    PerformanceConstants, Constants
)

logger = logging.getLogger("BabySleepSoundGenerator")


class NaturalSoundGenerator(SoundProfileGenerator):
    """Generator for natural sounds like heartbeats, shushing, and fan sounds."""
    
    def __init__(self, sample_rate: int, use_perlin: bool = True, seed: Optional[int] = None, **kwargs):
        """
        Initialize the natural sound generator.
        
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
        Generate a natural sound based on the specified type.
        
        Args:
            duration_seconds: Duration in seconds
            **kwargs: Additional parameters including sound_type and heartbeat_params
            
        Returns:
            Sound profile as numpy array
        """
        # Validate input parameters
        if duration_seconds <= 0:
            logger.error("Duration must be positive")
            raise ValueError("Duration must be positive")
        
        # Extract and validate sound type
        sound_type = kwargs.get('sound_type', 'heartbeat')
        if sound_type not in ['heartbeat', 'shushing', 'fan']:
            logger.error(f"Unknown natural sound type: {sound_type}")
            raise ValueError(f"Unknown natural sound type: {sound_type}")
        
        # Generator function mapping
        generators = {
            'heartbeat': self._generate_heartbeat,
            'shushing': self._generate_shushing_sound,
            'fan': self._generate_fan_sound
        }
        
        try:
            # Call the appropriate generator with error handling
            return generators[sound_type](duration_seconds, **kwargs)
        except Exception as e:
            logger.error(f"Error generating {sound_type} sound: {e}")
            # Generate a simple fallback sound
            return self._generate_fallback_sound(duration_seconds, sound_type)
    
    def _generate_fallback_sound(self, duration_seconds: int, sound_type: str) -> np.ndarray:
        """Generate a simple fallback sound if normal generation fails."""
        try:
            samples = int(duration_seconds * self.sample_rate)
            # Different fallbacks based on sound type
            if sound_type == 'heartbeat':
                return self._generate_simple_fallback_heartbeat(duration_seconds)
            elif sound_type == 'shushing':
                # Simple filtered noise for shushing
                white_noise = self.random_state.normal(0, 0.5, samples)
                b, a = signal.butter(2, [1500 / (self.sample_rate / 2), 4000 / (self.sample_rate / 2)], "band")
                return signal.lfilter(b, a, white_noise) * 0.7
            else:  # For fan or unknown
                # Simple filtered noise for fan
                white_noise = self.random_state.normal(0, 0.5, samples)
                b, a = signal.butter(2, 2000 / (self.sample_rate / 2), "low")
                return signal.lfilter(b, a, white_noise) * 0.6
        except Exception as e:
            logger.error(f"Error generating fallback sound: {e}")
            # Last resort: return simple noise
            return np.random.normal(0, 0.3, int(duration_seconds * self.sample_rate))
    
    def _generate_heartbeat(self, duration_seconds: int, **kwargs) -> np.ndarray:
        """
        Generate a maternal heartbeat sound with natural variations.
        
        Args:
            duration_seconds: Duration in seconds
            **kwargs: Additional parameters including heartbeat_params
            
        Returns:
            Heartbeat audio array
        """
        # Extract and validate heartbeat parameters
        heartbeat_params = kwargs.get('heartbeat_params', None)
        if heartbeat_params is None:
            heartbeat_params = HeartbeatParameters()
        elif not isinstance(heartbeat_params, HeartbeatParameters):
            logger.warning("Invalid heartbeat parameters, using defaults")
            heartbeat_params = HeartbeatParameters()
        
        # Validate BPM range
        if heartbeat_params.variable_rate:
            min_bpm, max_bpm = heartbeat_params.bpm_range
            if min_bpm <= 0 or max_bpm <= 0 or min_bpm >= max_bpm:
                logger.warning("Invalid BPM range, using defaults")
                heartbeat_params.bpm_range = (60.0, 80.0)
        
        try:
            # Memory-efficient array allocation
            samples = int(duration_seconds * self.sample_rate)
            heartbeat = np.zeros(samples, dtype=np.float32)  # Use float32 to reduce memory usage

            # Generate beat times based on variable or fixed rate
            beat_samples = self._generate_beat_samples(
                duration_seconds, 
                heartbeat_params.variable_rate,
                heartbeat_params.base_bpm,
                heartbeat_params.bpm_range,
                heartbeat_params.variation_period_minutes
            )
            
            # Generate amplitude variations
            amp_variations = self._generate_amplitude_variations(
                len(beat_samples), 
                HeartbeatConstants.AMPLITUDE_VARIATION_PCT
            )

            # Create the heartbeat waveform
            self._create_heartbeat_waveform(
                heartbeat, 
                beat_samples,
                amp_variations,
                HeartbeatConstants.LUB_DURATION_SECONDS,
                HeartbeatConstants.DUB_DURATION_SECONDS,
                HeartbeatConstants.DUB_DELAY_SECONDS,
                HeartbeatConstants.LUB_FREQUENCY_HZ,
                HeartbeatConstants.DUB_FREQUENCY_HZ
            )

            # Apply a subtle low-pass filter for more natural sound
            b, a = signal.butter(4, 200 / (self.sample_rate / 2), "low")
            heartbeat = signal.lfilter(b, a, heartbeat)

            # Normalize and sanitize
            max_val = np.max(np.abs(heartbeat))
            if max_val > 0:
                heartbeat = heartbeat / max_val * HeartbeatConstants.DEFAULT_AMPLITUDE

            return self.sanitize_audio(heartbeat, max_amplitude=HeartbeatConstants.DEFAULT_AMPLITUDE)
            
        except Exception as e:
            logger.error(f"Error generating heartbeat: {e}")
            # Generate a simple fallback heartbeat in case of error
            return self._generate_simple_fallback_heartbeat(duration_seconds)
    
    def _generate_beat_samples(
        self, duration_seconds: int, variable_rate: bool, 
        base_bpm: float, bpm_range: Tuple[float, float], 
        variation_period_minutes: float
    ) -> np.ndarray:
        """
        Generate sample indices for heartbeat timings.
        
        Args:
            duration_seconds: Duration in seconds
            variable_rate: Whether to use variable heart rate
            base_bpm: Base BPM value
            bpm_range: Range of BPM variation (min, max)
            variation_period_minutes: Period of variation in minutes
            
        Returns:
            Array of sample indices for heartbeats
        """
        samples = int(duration_seconds * self.sample_rate)
        
        if not variable_rate:
            # Fixed BPM, simple calculation
            frequency = base_bpm / 60.0
            beat_period = 1.0 / frequency
            return (np.arange(0, duration_seconds, beat_period) * self.sample_rate).astype(int)
        
        # For variable rate, we need to compute instantaneous BPM at each time point
        if HAS_PERLIN and self.use_perlin:
            # Generate smooth perlin noise for natural BPM variations
            perlin_duration = duration_seconds / 10  # Generate shorter noise and stretch
            bpm_noise = generate_perlin_noise(
                self.sample_rate, 
                perlin_duration, 
                octaves=1, 
                persistence=0.5,
                seed=self.random_state.seed
            )
            
            # Stretch the noise to match desired variation period
            cycles = duration_seconds / (variation_period_minutes * 60)
            indices = np.linspace(0, len(bpm_noise) - 1, int(cycles * 100) + 1)
            indices = np.clip(indices.astype(int), 0, len(bpm_noise) - 1)
            
            # Scale to desired BPM range
            min_bpm, max_bpm = bpm_range
            bpm_range_halfwidth = (max_bpm - min_bpm) / 2
            center_bpm = min_bpm + bpm_range_halfwidth
            bpm_variations = center_bpm + bpm_range_halfwidth * np.clip(bpm_noise[indices], -1, 1)
            
            # Interpolate to full duration
            t_values = np.linspace(0, cycles, len(bpm_variations))
            t_points = np.linspace(0, cycles, int(duration_seconds) + 1)
            bpm_curve = np.interp(t_points, t_values, bpm_variations)
        else:
            # Use sinusoidal variation as fallback
            variation_freq = 1 / (variation_period_minutes * 60)  # Hz
            t = np.linspace(0, duration_seconds, int(duration_seconds) + 1)
            min_bpm, max_bpm = bpm_range
            center_bpm = (min_bpm + max_bpm) / 2
            bpm_range_halfwidth = (max_bpm - min_bpm) / 2
            bpm_curve = center_bpm + bpm_range_halfwidth * np.sin(2 * np.pi * variation_freq * t)
        
        # Integrate the instantaneous frequency to get beat times
        beat_times = [0]  # Start with first beat at t=0
        current_time = 0
        
        while current_time < duration_seconds:
            # Get BPM at current time and convert to period
            current_bpm = np.interp(current_time, np.arange(len(bpm_curve)), bpm_curve)
            period = 60 / current_bpm  # seconds per beat
            
            # Next beat time
            current_time += period
            if current_time < duration_seconds:
                beat_times.append(current_time)
        
        # Convert to sample indices
        return (np.array(beat_times) * self.sample_rate).astype(int)
    
    def _generate_amplitude_variations(self, num_beats: int, variation_pct: float) -> np.ndarray:
        """
        Generate natural amplitude variations for heartbeats.
        
        Args:
            num_beats: Number of beats to generate variations for
            variation_pct: Percentage of amplitude variation
            
        Returns:
            Array of amplitude variation factors
        """
        if HAS_PERLIN and self.use_perlin:
            # Generate very slow perlin noise for natural amplitude variations
            amp_variation = generate_perlin_noise(
                self.sample_rate, 
                num_beats / 10,  # Short duration, will be stretched 
                octaves=1, 
                persistence=0.5,
                seed=self.random_state.seed
            )
            
            # Stretch to get only a few variations over all beats
            indices = np.linspace(0, len(amp_variation) - 1, num_beats)
            indices = np.clip(indices.astype(int), 0, len(amp_variation) - 1)
            amp_variations = 0.85 + variation_pct * np.clip(amp_variation[indices], -1, 1)
        else:
            # Fallback to random variations with smoothing
            amp_variations = 0.85 + variation_pct * self.random_state.normal(0, 0.3, num_beats)
            
            # Smooth the variations with a moving average
            kernel_size = min(5, num_beats)
            if kernel_size > 0:
                kernel = np.ones(kernel_size) / kernel_size
                amp_variations = np.convolve(amp_variations, kernel, mode="same")
        
        return amp_variations
    
    def _create_heartbeat_waveform(
        self, 
        heartbeat: np.ndarray, 
        beat_samples: np.ndarray,
        amp_variations: np.ndarray,
        lub_duration: float,
        dub_duration: float,
        dub_delay: float,
        lub_frequency: float,
        dub_frequency: float
    ) -> None:
        """
        Create the actual heartbeat waveform with lub and dub sounds.
        
        Args:
            heartbeat: Output array to fill with heartbeat
            beat_samples: Array of beat start sample indices
            amp_variations: Array of amplitude variation factors
            lub_duration: Duration of the 'lub' sound in seconds
            dub_duration: Duration of the 'dub' sound in seconds
            dub_delay: Delay after 'lub' before 'dub' starts, in seconds
            lub_frequency: Frequency of the 'lub' sound in Hz
            dub_frequency: Frequency of the 'dub' sound in Hz
        """
        samples = len(heartbeat)
        lub_samples = int(lub_duration * self.sample_rate)
        dub_samples = int(dub_duration * self.sample_rate)
        dub_delay_samples = int(dub_delay * self.sample_rate)
        
        # Pre-compute common envelopes and waveforms to reduce redundant calculations
        lub_env_width = lub_duration / 5
        dub_env_width = dub_duration / 5
        
        # Create time vectors for lub and dub waveforms
        t_lub = np.linspace(0, lub_duration, lub_samples, endpoint=False)
        t_dub = np.linspace(0, dub_duration, dub_samples, endpoint=False)
        
        # Create envelopes
        lub_envelope = np.exp(-(t_lub**2) / (2 * lub_env_width**2))
        dub_envelope = np.exp(-(t_dub**2) / (2 * dub_env_width**2))
        
        # Create base waveforms
        lub_wave = np.sin(2 * np.pi * lub_frequency * t_lub)
        dub_wave = np.sin(2 * np.pi * dub_frequency * t_dub)
        
        # Modulate with envelopes
        lub_template = lub_envelope * lub_wave
        dub_template = 0.7 * dub_envelope * dub_wave  # Dub is softer
        
        # Apply to each beat with careful bounds checking
        for i, beat_start_sample in enumerate(beat_samples):
            if beat_start_sample >= samples:
                continue
                
            # Apply amplitude variation for this beat
            amp_factor = amp_variations[min(i, len(amp_variations) - 1)]
            
            # "Lub" part
            lub_end = min(beat_start_sample + lub_samples, samples)
            lub_len = lub_end - beat_start_sample
            if lub_len > 0:
                heartbeat[beat_start_sample:lub_end] += lub_template[:lub_len] * amp_factor
            
            # "Dub" part (follows the lub)
            dub_start_sample = min(beat_start_sample + dub_delay_samples, samples - 1)
            dub_end = min(dub_start_sample + dub_samples, samples)
            dub_len = dub_end - dub_start_sample
            
            if dub_len > 0 and dub_start_sample < samples:
                heartbeat[dub_start_sample:dub_end] += dub_template[:dub_len] * amp_factor
    
    def _generate_simple_fallback_heartbeat(self, duration_seconds: int) -> np.ndarray:
        """Generate simple heartbeat as fallback in case of error."""
        try:
            samples = int(duration_seconds * self.sample_rate)
            heartbeat = np.zeros(samples, dtype=np.float32)
            
            # Simple constant rate heartbeat at 70 BPM
            bpm = 70
            period = 60 / bpm  # seconds per beat
            beat_interval_samples = int(period * self.sample_rate)
            
            # Vectorized approach for efficiency
            beat_positions = np.arange(0, samples, beat_interval_samples)
            
            for pos in beat_positions:
                if pos + 200 < samples:  # 200 samples for lub
                    t = np.linspace(0, 0.1, 200)
                    heartbeat[pos:pos+200] = 0.7 * np.sin(2 * np.pi * 60 * t) * np.sin(np.linspace(0, np.pi, 200))**2
                
                if pos + 500 < samples and pos + 700 < samples:  # dub after 0.25s
                    t = np.linspace(0, 0.1, 200)
                    heartbeat[pos+500:pos+700] = 0.4 * np.sin(2 * np.pi * 45 * t) * np.sin(np.linspace(0, np.pi, 200))**2
            
            return self.sanitize_audio(heartbeat, max_amplitude=0.7)
                
        except Exception as e:
            logger.error(f"Error generating fallback heartbeat: {e}")
            # Return simple pulse waveform as last resort
            samples = int(duration_seconds * self.sample_rate)
            pulse_rate = 1.2  # beats per second
            return 0.5 * np.sin(2 * np.pi * pulse_rate * np.linspace(0, duration_seconds, samples))

    def _generate_shushing_sound(self, duration_seconds: int, **kwargs) -> np.ndarray:
        """
        Generate a more natural shushing sound similar to what parents do to calm babies
        
        Args:
            duration_seconds: Length of the sound in seconds
            **kwargs: Additional parameters
            
        Returns:
            Shushing sound array
        """
        try:
            samples = int(duration_seconds * self.sample_rate)

            # Start with filtered white noise
            white_noise = self.random_state.normal(0, 0.5, samples)

            # Apply bandpass filter to focus on "shh" frequencies
            b, a = signal.butter(
                4, 
                [ShushingConstants.BANDPASS_LOW_HZ / (self.sample_rate / 2), 
                 ShushingConstants.BANDPASS_HIGH_HZ / (self.sample_rate / 2)], 
                "band"
            )
            shushing = signal.lfilter(b, a, white_noise)

            # Generate modulation curve for natural rhythm
            shush_modulation = self._generate_shushing_modulation(
                samples,
                duration_seconds,
                ShushingConstants.SHUSH_RATE_PER_SECOND,
                ShushingConstants.MODULATION_MIN,
                ShushingConstants.MODULATION_MAX
            )

            # Apply the modulation
            shushing = shushing * shush_modulation

            # Apply subtle formant filtering to make it sound more like human shushing
            formant_filter = signal.firwin2(
                101,
                [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                [0.1, 0.2, 0.5, 1.0, 0.8, 0.6, 0.7, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1, 0.05]
            )
            shushing = signal.lfilter(formant_filter, 1, shushing)

            # Normalize and sanitize
            return self.sanitize_audio(shushing, max_amplitude=ShushingConstants.DEFAULT_AMPLITUDE)

        except Exception as e:
            logger.error(f"Error generating shushing sound: {e}")
            # Return simple filtered noise as fallback
            samples = int(duration_seconds * self.sample_rate)
            noise = self.random_state.normal(0, 0.5, samples)
            try:
                # Simple bandpass filter
                b, a = signal.butter(2, [1500 / (self.sample_rate / 2), 4000 / (self.sample_rate / 2)], "band")
                filtered = signal.lfilter(b, a, noise)
                return self.sanitize_audio(filtered, max_amplitude=0.7)
            except:
                return noise * 0.7
    
    def _generate_shushing_modulation(
        self, samples: int, duration_seconds: float, 
        shush_rate: float, min_level: float, max_level: float
    ) -> np.ndarray:
        """
        Generate a natural shushing modulation pattern.
        
        Args:
            samples: Number of samples to generate
            duration_seconds: Duration in seconds
            shush_rate: Shushing cycles per second
            min_level: Minimum modulation level
            max_level: Maximum modulation level
            
        Returns:
            Modulation curve array
        """
        t = np.linspace(0, duration_seconds, samples, endpoint=False)
        
        if HAS_PERLIN and self.use_perlin:
            try:
                # Generate slow perlin noise for organic variations
                perlin_duration = duration_seconds / 5  # Short duration, will be stretched
                shush_rhythm = generate_perlin_noise(
                    self.sample_rate,
                    perlin_duration, 
                    octaves=2, 
                    persistence=0.6,
                    seed=self.random_state.seed
                )
                
                # Scale to appropriate rhythm rate
                indices = np.linspace(0, len(shush_rhythm) - 1, samples)
                indices = np.clip(indices.astype(int), 0, len(shush_rhythm) - 1)
                
                # Transform to min-max range
                level_range = max_level - min_level
                shush_modulation = min_level + level_range * (0.5 + 0.5 * np.clip(shush_rhythm[indices], -1, 1))
                
                return shush_modulation
            except Exception as e:
                logger.warning(f"Error creating perlin modulation for shushing: {e}")
                # Fall through to sine-based fallback
        
        # Fallback to sine-based rhythm with some randomness
        phase_variation = 0.2 * self.random_state.normal(0, 1, int(duration_seconds * shush_rate) + 1)
        
        # Initialize modulation array
        shush_modulation = np.zeros(samples, dtype=np.float32)
        
        # Create a modified sine wave with phase variations
        for i in range(int(duration_seconds * shush_rate)):
            start_idx = int(i * self.sample_rate / shush_rate)
            end_idx = int((i + 1) * self.sample_rate / shush_rate)
            end_idx = min(end_idx, samples)
            
            # Create a single shush cycle with natural attack and decay
            cycle_len = end_idx - start_idx
            if cycle_len > 0:
                cycle = (
                    min_level + 
                    (max_level - min_level) * 
                    np.sin(np.linspace(0 + phase_variation[i], np.pi + phase_variation[i], cycle_len)) ** 2
                )
                shush_modulation[start_idx:end_idx] = cycle
        
        return shush_modulation

    def _generate_fan_sound(self, duration_seconds: int, **kwargs) -> np.ndarray:
        """
        Generate a more realistic fan or air conditioner type sound
        
        Args:
            duration_seconds: Length of the sound in seconds
            **kwargs: Additional parameters
            
        Returns:
            Fan sound array
        """
        try:
            samples = int(duration_seconds * self.sample_rate)

            # Generate the base noise (mix of pink and white)
            base_noise = self._generate_fan_base_noise(samples)
            
            # Generate the fan rotation modulation
            rotation_mod = self._generate_fan_rotation_modulation(
                samples, 
                duration_seconds,
                FanConstants.BASE_ROTATION_HZ,
                FanConstants.SPEED_VARIATION_PCT
            )
            
            # Apply fan modulation
            fan_sound = base_noise * (1 + rotation_mod)

            # Apply resonance filtering to simulate fan housing
            fan_sound = self._apply_fan_resonance(
                fan_sound, 
                FanConstants.RESONANCE_FREQUENCIES_HZ,
                FanConstants.RESONANCE_Q_FACTOR
            )

            # Apply final bandpass filter to shape the spectrum
            b, a = signal.butter(
                3, 
                [FanConstants.BANDPASS_LOW_HZ / (self.sample_rate / 2), 
                 FanConstants.BANDPASS_HIGH_HZ / (self.sample_rate / 2)], 
                "band"
            )
            fan_sound = signal.lfilter(b, a, fan_sound)

            # Normalize and sanitize
            return self.sanitize_audio(fan_sound, max_amplitude=FanConstants.DEFAULT_AMPLITUDE)

        except Exception as e:
            logger.error(f"Error generating fan sound: {e}")
            # Return simple filtered noise as fallback
            samples = int(duration_seconds * self.sample_rate)
            noise = self.random_state.normal(0, 0.5, samples)
            try:
                # Apply a simple low-pass filter to approximate fan sound
                b, a = signal.butter(2, 1500 / (self.sample_rate / 2), "low")
                filtered = signal.lfilter(b, a, noise)
                
                # Add simple modulation
                t = np.linspace(0, duration_seconds, samples)
                modulation = 1 + 0.1 * np.sin(2 * np.pi * 4 * t)  # 4 Hz fan blade rotation
                
                result = filtered * modulation
                return self.sanitize_audio(result, max_amplitude=0.7)
            except:
                return noise * 0.6

    def _generate_fan_base_noise(self, samples: int) -> np.ndarray:
        """
        Generate the base noise for a fan sound (mix of white and pink).
        
        Args:
            samples: Number of samples to generate
            
        Returns:
            Base noise array
        """
        # Start with white noise
        white = self.random_state.normal(0, 0.5, samples)
        
        # Create pink noise using simple filtering
        pink_filter = signal.firwin(1001, 1000 / (self.sample_rate / 2), window='hann')
        pink = signal.lfilter(pink_filter, 1, white)
        
        # Normalize pink component
        max_val = np.max(np.abs(pink))
        if max_val > 0:
            pink = pink / max_val * 0.5
        
        # Mix white and pink noise
        return white * 0.7 + pink * 0.3
    
    def _generate_fan_rotation_modulation(
        self, samples: int, duration_seconds: float, 
        base_rotation_hz: float, speed_variation_pct: float
    ) -> np.ndarray:
        """
        Generate the rotation modulation for a fan sound.
        
        Args:
            samples: Number of samples to generate
            duration_seconds: Duration in seconds
            base_rotation_hz: Base rotation frequency in Hz
            speed_variation_pct: Percentage of speed variation
            
        Returns:
            Rotation modulation array
        """
        t = np.linspace(0, duration_seconds, samples, endpoint=False)
        
        if HAS_PERLIN and self.use_perlin:
            try:
                # Use perlin noise for natural speed variations
                speed_variation = generate_perlin_noise(
                    self.sample_rate,
                    duration_seconds / 10, 
                    octaves=1, 
                    persistence=0.5,
                    seed=self.random_state.seed
                )
                
                # Stretch to get slow variations over the duration
                indices = np.linspace(0, len(speed_variation) - 1, samples)
                indices = np.clip(indices.astype(int), 0, len(speed_variation) - 1)
                
                # Modulate the rotation speed
                rotation_speed = base_rotation_hz * (1 + speed_variation_pct * speed_variation[indices])
                
                # Integrate speed to get phase
                phase = np.cumsum(rotation_speed) / self.sample_rate * 2 * np.pi
                
                # Calculate modulation with harmonic overtones
                rotation_modulation = (0.08 * np.sin(phase) + 
                                    0.04 * np.sin(2 * phase) + 
                                    0.02 * np.sin(3 * phase))
                
                return rotation_modulation
            except Exception as e:
                logger.warning(f"Error creating perlin modulation for fan: {e}")
                # Fall through to sine-based fallback
        
        # Fallback to simpler fan simulation
        # Add a slight drift to the rotation speed
        drift = speed_variation_pct * np.sin(2 * np.pi * 0.05 * t)
        rotation_modulation = 0.1 * np.sin(2 * np.pi * base_rotation_hz * (1 + drift) * t)
        
        # Add some harmonics
        rotation_modulation += 0.05 * np.sin(2 * 2 * np.pi * base_rotation_hz * (1 + drift) * t)
        
        return rotation_modulation
    
    def _apply_fan_resonance(
        self, audio: np.ndarray, resonance_frequencies: List[float], q_factor: float
    ) -> np.ndarray:
        """
        Apply resonance filtering to simulate fan housing.
        
        Args:
            audio: Audio array to filter
            resonance_frequencies: List of resonance frequencies in Hz
            q_factor: Q factor for resonance filters
            
        Returns:
            Filtered audio array
        """
        # Apply each resonance filter sequentially
        filtered = audio.copy()
        for freq in resonance_frequencies:
            try:
                b, a = signal.iirpeak(freq, q_factor, self.sample_rate)
                filtered = signal.lfilter(b, a, filtered)
            except Exception as e:
                logger.warning(f"Error applying resonance filter at {freq}Hz: {e}")
                continue
                
        return filtered

    def apply_dynamic_shushing(
        self, audio: np.ndarray, shushing: np.ndarray, params: DynamicShushing
    ) -> np.ndarray:
        """
        Apply cry-responsive amplitude modulation to shushing sound.
        This mimics how parents naturally increase shushing volume when a baby cries.

        Args:
            audio: Main audio array
            shushing: Shushing sound array (same length as audio)
            params: Parameters for dynamic shushing

        Returns:
            Modified audio with dynamic shushing
        """
        if not params or not params.enabled:
            return audio

        try:
            # Parameter validation
            if not isinstance(params, DynamicShushing):
                logger.warning("Invalid dynamic shushing parameters")
                return audio
            
            # Extract and validate parameters
            base_level_db = max(-60, min(0, params.base_level_db))
            cry_response_db = max(0, min(20, params.cry_response_db))
            response_time_ms = max(10, min(1000, params.response_time_ms))

            # Convert dB difference to amplitude ratio
            level_increase = 10 ** (cry_response_db / 20.0)

            # Response time in samples
            response_samples = int(response_time_ms * self.sample_rate / 1000)

            # Simulate cry periods using perlin noise
            cry_envelope = self._generate_cry_periods(
                len(audio), response_samples, level_increase
            )

            # Apply envelope to shushing efficiently
            dynamic_shushing = apply_modulation(shushing, cry_envelope)

            # Ensure compatible shapes for mixing
            is_stereo = len(audio.shape) > 1
            if is_stereo and len(dynamic_shushing.shape) == 1:
                dynamic_shushing_stereo = np.zeros_like(audio)
                for c in range(audio.shape[1]):
                    dynamic_shushing_stereo[:, c] = dynamic_shushing
                dynamic_shushing = dynamic_shushing_stereo
            
            # Mix with original audio
            output = audio * 0.7 + dynamic_shushing * 0.3

            # Normalize if needed
            max_val = np.max(np.abs(output))
            if max_val > Constants.MAX_AUDIO_VALUE:
                output = output / max_val * Constants.MAX_AUDIO_VALUE

            return output

        except Exception as e:
            logger.error(f"Error applying dynamic shushing: {e}")
            # Return mixed audio without dynamic modulation as fallback
            try:
                is_stereo = len(audio.shape) > 1
                if is_stereo:
                    # Ensure shushing is stereo before mixing
                    if len(shushing.shape) == 1:
                        shushing_stereo = np.zeros_like(audio)
                        for c in range(audio.shape[1]):
                            shushing_stereo[:, c] = shushing
                        output = audio * 0.7 + shushing_stereo * 0.3
                    else:
                        output = audio * 0.7 + shushing * 0.3
                else:
                    output = audio * 0.7 + shushing * 0.3
                    
                return self.sanitize_audio(output)
            except:
                # Return original audio if mixing fails
                return audio
    
    def _generate_cry_periods(
        self, samples: int, response_samples: int, level_increase: float
    ) -> np.ndarray:
        """
        Generate simulated cry periods for dynamic shushing.
        
        Args:
            samples: Number of samples to generate
            response_samples: Response time in samples
            level_increase: Level increase factor during crying
            
        Returns:
            Envelope for cry-responsive shushing
        """
        # Create output envelope starting at base level
        envelope = np.ones(samples, dtype=np.float32)
        
        # At least 5 minutes between potential cry periods
        interval_min_samples = 5 * 60 * self.sample_rate
        
        if HAS_PERLIN and self.use_perlin:
            try:
                # Generate very slow perlin noise for cry likelihood
                cry_likelihood = generate_perlin_noise(
                    self.sample_rate,
                    samples / self.sample_rate / 10, 
                    octaves=1, 
                    persistence=0.5,
                    seed=self.random_state.seed
                )
                
                # Stretch to audio length
                indices = np.linspace(0, len(cry_likelihood) - 1, samples)
                indices = np.clip(indices.astype(int), 0, len(cry_likelihood) - 1)
                cry_likelihood = cry_likelihood[indices]
                
                # Find potential cry starts (threshold crossings)
                cry_threshold = 0.6
                crossing_points = np.where(
                    np.diff((cry_likelihood > cry_threshold).astype(int)) > 0
                )[0]
                
                # Filter to ensure minimum spacing
                if len(crossing_points) > 0:
                    filtered_points = [crossing_points[0]]
                    for point in crossing_points[1:]:
                        if point - filtered_points[-1] >= interval_min_samples:
                            filtered_points.append(point)
                    
                    # Create cry periods
                    for start in filtered_points:
                        if start < samples - interval_min_samples:
                            duration = self.random_state.randint(30, 60) * self.sample_rate
                            end = min(start + duration, samples)
                            
                            # Apply ramp up at start of cry
                            ramp_up_end = min(start + response_samples, samples)
                            if ramp_up_end > start:
                                ramp_up = np.linspace(1.0, level_increase, ramp_up_end - start)
                                envelope[start:ramp_up_end] = ramp_up
                            
                            # Sustain level during cry
                            sustain_end = max(ramp_up_end, min(end - response_samples, samples))
                            if sustain_end > ramp_up_end:
                                envelope[ramp_up_end:sustain_end] = level_increase
                            
                            # Ramp down at end of cry
                            if sustain_end < samples:
                                ramp_down_end = min(sustain_end + response_samples, samples)
                                if ramp_down_end > sustain_end:
                                    ramp_down = np.linspace(level_increase, 1.0, ramp_down_end - sustain_end)
                                    envelope[sustain_end:ramp_down_end] = ramp_down
            else:
                # Use regular intervals if no Perlin noise
                self._generate_regular_cry_periods(
                    envelope, samples, response_samples, level_increase, interval_min_samples
                )
                
            return envelope
                
        except Exception as e:
            logger.warning(f"Error generating perlin-based cry periods: {e}")
            # Fall back to regular intervals
            self._generate_regular_cry_periods(
                envelope, samples, response_samples, level_increase, interval_min_samples
            )
            return envelope
    
    def _generate_regular_cry_periods(
        self, envelope: np.ndarray, samples: int, 
        response_samples: int, level_increase: float, interval_min_samples: int
    ) -> None:
        """
        Generate regularly spaced cry periods as fallback.
        
        Args:
            envelope: Envelope array to fill
            samples: Total number of samples
            response_samples: Response time in samples
            level_increase: Level increase factor during crying
            interval_min_samples: Minimum interval between cries in samples
        """
        interval_samples = 8 * 60 * self.sample_rate  # 8 minutes
        for start in range(0, samples, interval_samples):
            if start < samples - interval_min_samples:
                duration = self.random_state.randint(30, 60) * self.sample_rate
                end = min(start + duration, samples)
                
                # Apply ramp up at start of cry
                ramp_up_end = min(start + response_samples, samples)
                if ramp_up_end > start:
                    ramp_up = np.linspace(1.0, level_increase, ramp_up_end - start)
                    envelope[start:ramp_up_end] = ramp_up
                
                # Sustain level during cry
                sustain_end = max(ramp_up_end, min(end - response_samples, samples))
                if sustain_end > ramp_up_end:
                    envelope[ramp_up_end:sustain_end] = level_increase
                
                # Ramp down at end of cry
                if sustain_end < samples:
                    ramp_down_end = min(sustain_end + response_samples, samples)
                    if ramp_down_end > sustain_end:
                        ramp_down = np.linspace(level_increase, 1.0, ramp_down_end - sustain_end)
                        envelope[sustain_end:ramp_down_end] = ramp_down

    def apply_parental_voice(
        self, audio: np.ndarray, params: ParentalVoice
    ) -> np.ndarray:
        """
        Add a simulated parental voice hum at a very low level.
        This is based on research showing familiar voices aid sleep in infants.

        Args:
            audio: Input audio array
            params: Parameters for parental voice

        Returns:
            Audio with subtle parental voice hum
        """
        if not params or not params.enabled:
            return audio

        try:
            # Validate parameters
            if not isinstance(params, ParentalVoice):
                logger.warning("Invalid parental voice parameters")
                return audio
            
            # Extract and validate parameters
            mix_level_db = max(-60, min(-6, params.mix_level_db))

            # Convert dB to linear scale
            mix_ratio = 10 ** (mix_level_db / 20.0)

            # Get audio dimensions
            samples = len(audio)
            is_stereo = len(audio.shape) > 1
            channels = audio.shape[1] if is_stereo else 1

            # Generate the humming sound
            hum_output = self._generate_parental_humming(
                samples, channels, is_stereo
            )

            # Mix with original audio at specified level
            if is_stereo:
                output = audio * (1 - mix_ratio) + hum_output * mix_ratio
            else:
                output = audio * (1 - mix_ratio) + hum_output * mix_ratio

            return self.sanitize_audio(output)

        except Exception as e:
            logger.error(f"Error applying parental voice: {e}")
            # Return original audio if voice processing fails
            return audio
    
    def _generate_parental_humming(
        self, samples: int, channels: int, is_stereo: bool
    ) -> np.ndarray:
        """
        Generate a parental humming sound.
        
        Args:
            samples: Number of samples to generate
            channels: Number of audio channels
            is_stereo: Whether the output should be stereo
            
        Returns:
            Humming sound array
        """
        # Create output array with appropriate shape
        if is_stereo:
            hum_output = np.zeros((samples, channels), dtype=np.float32)
        else:
            hum_output = np.zeros(samples, dtype=np.float32)
            
        # Basic parameters for a natural-sounding hum
        hum_freq = 220.0  # Typical female hum, ~A3
        hum_duration_seconds = 2.0
        hum_interval_seconds = 8.0  # Space between hums

        # Convert to samples
        hum_samples = int(hum_duration_seconds * self.sample_rate)
        interval_samples = int(hum_interval_seconds * self.sample_rate)

        # Time vector for a single hum
        t_hum = np.linspace(0, hum_duration_seconds, hum_samples, endpoint=False)

        # Generate basic hum with harmonics for realism
        hum_base = np.sin(2 * np.pi * hum_freq * t_hum)
        hum_harmonic1 = 0.5 * np.sin(2 * np.pi * hum_freq * 2 * t_hum)  # First harmonic
        hum_harmonic2 = 0.3 * np.sin(2 * np.pi * hum_freq * 3 * t_hum)  # Second harmonic
        hum_harmonic3 = 0.1 * np.sin(2 * np.pi * hum_freq * 4 * t_hum)  # Third harmonic

        basic_hum = hum_base + hum_harmonic1 + hum_harmonic2 + hum_harmonic3

        # Apply envelope for natural attack and decay
        attack_samples = int(0.2 * hum_samples)  # 20% attack
        sustain_samples = int(0.5 * hum_samples)  # 50% sustain
        decay_samples = hum_samples - attack_samples - sustain_samples  # 30% decay

        # Create envelope segments
        envelope = np.zeros_like(basic_hum)
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        envelope[attack_samples:attack_samples+sustain_samples] = 1.0
        envelope[attack_samples+sustain_samples:] = np.linspace(1, 0, decay_samples)

        # Apply envelope
        hum = basic_hum * envelope

        # Apply vocal formant filtering for more realism
        b, a = signal.butter(2, [400 / (self.sample_rate / 2), 2000 / (self.sample_rate / 2)], "band")
        hum = signal.lfilter(b, a, hum)

        # Place hums at regular intervals
        cycle_length = hum_samples + interval_samples
        num_cycles = (samples // cycle_length) + 1

        for i in range(num_cycles):
            start_idx = i * cycle_length
            end_idx = min(start_idx + hum_samples, samples)
            hum_len = end_idx - start_idx

            if hum_len > 0:
                if is_stereo:
                    for c in range(channels):
                        hum_output[start_idx:end_idx, c] = hum[:hum_len]
                else:
                    hum_output[start_idx:end_idx] = hum[:hum_len]

        return hum_output