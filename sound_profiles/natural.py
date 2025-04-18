"""
Natural sound generators like heartbeat, shushing, and fan sounds.
"""

import numpy as np
from scipy import signal
import logging
from typing import Optional, Dict, Any, Union

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
        # Get sound type with default to "heartbeat"
        sound_type = kwargs.get('sound_type', 'heartbeat')
        
        try:
            if sound_type == 'heartbeat':
                heartbeat_params = kwargs.get('heartbeat_params', None)
                return self.generate_heartbeat(duration_seconds, heartbeat_params)
            elif sound_type == 'shushing':
                return self.generate_shushing_sound(duration_seconds)
            elif sound_type == 'fan':
                return self.generate_fan_sound(duration_seconds)
            else:
                raise ValueError(f"Unknown natural sound type: {sound_type}")
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
    
    def generate_heartbeat(
        self,
        duration_seconds: int,
        heartbeat_params: Optional[HeartbeatParameters] = None,
    ) -> np.ndarray:
        """
        Generate a maternal heartbeat sound with natural variations.

        Args:
            duration_seconds: Duration in seconds
            heartbeat_params: Parameters for heartbeat generation

        Returns:
            Heartbeat audio array
        """
        # Use default parameters if none provided
        if heartbeat_params is None:
            heartbeat_params = HeartbeatParameters()
            
        try:
            # Extract parameters
            variable_rate = heartbeat_params.variable_rate
            bpm = heartbeat_params.base_bpm
            bpm_range = heartbeat_params.bpm_range
            variation_period_minutes = heartbeat_params.variation_period_minutes
            
            samples = int(duration_seconds * self.sample_rate)

            # Time points
            t = np.linspace(0, duration_seconds, samples, endpoint=False)

            # Create the base heartbeat using a combination of sine waves
            heartbeat = np.zeros(samples)

            # For variable rate, we'll use either perlin noise or sine modulation
            if variable_rate:
                # Calculate how many variation cycles there will be
                cycles = duration_seconds / (variation_period_minutes * 60)

                if HAS_PERLIN and self.use_perlin:
                    # Generate smooth perlin noise for natural BPM variations
                    bpm_noise = generate_perlin_noise(
                        self.sample_rate, 
                        duration_seconds / 10, 
                        octaves=1, 
                        persistence=0.5,
                        seed=self.random_state.seed
                    )
                    
                    # Stretch the noise to match desired cycle period
                    indices = np.linspace(0, len(bpm_noise) - 1, int(cycles * 100) + 1)
                    indices = np.clip(indices.astype(int), 0, len(bpm_noise) - 1)
                    bpm_variations = bpm_noise[indices]
                    
                    # Interpolate to full length - FIXED: Use arrays of same length
                    full_indices = np.linspace(0, len(bpm_variations) - 1, int(duration_seconds) + 1)
                    index_array = np.arange(len(bpm_variations))
                    if len(index_array) > 0:
                        bpm_variations = np.interp(full_indices, index_array, bpm_variations)
                    
                    # Scale to desired BPM range
                    min_bpm, max_bpm = bpm_range
                    bpm_range_halfwidth = (max_bpm - min_bpm) / 2
                    center_bpm = min_bpm + bpm_range_halfwidth
                    
                    # Map from [-0.5, 0.5] to [min_bpm, max_bpm]
                    bpm_variations = center_bpm + bpm_range_halfwidth * bpm_variations
                else:
                    # Use sinusoidal variation as fallback
                    variation_freq = 1 / (variation_period_minutes * 60)  # Hz
                    min_bpm, max_bpm = bpm_range
                    center_bpm = (min_bpm + max_bpm) / 2
                    bpm_range_halfwidth = (max_bpm - min_bpm) / 2
                    bpm_variations = center_bpm + bpm_range_halfwidth * np.sin(2 * np.pi * variation_freq * t)

                # Now we have a time-varying BPM function
                # We'll integrate it to get the total beats over time
                # This gives us the phase of the heartbeat at each point in time
                
                # Start with constant period for the first beat
                instantaneous_period = 60 / bpm_variations[0]  # seconds per beat
                beat_times = [0]  # Start time of each beat

                # Calculate when each beat should occur based on variable BPM
                current_time = 0
                beat_index = 0

                while current_time < duration_seconds:
                    # Get BPM at this time point - FIXED: Make sure arrays used in np.interp have the same length
                    # Create t_values appropriate for the length of bpm_variations
                    t_values = np.linspace(0, duration_seconds, len(bpm_variations))
                    current_bpm = np.interp(current_time, t_values, bpm_variations)
                    
                    # Convert to period (seconds per beat)
                    instantaneous_period = 60 / current_bpm
                    # Next beat time
                    current_time += instantaneous_period
                    if current_time < duration_seconds:
                        beat_times.append(current_time)
                        beat_index += 1

                # Convert beat times to sample indices
                beat_samples = (np.array(beat_times) * self.sample_rate).astype(int)
            else:
                # Fixed BPM, simple calculation
                # Convert bpm to frequency in Hz
                frequency = bpm / 60.0
                beat_period = 1.0 / frequency
                beat_samples = (np.arange(0, duration_seconds, beat_period) * self.sample_rate).astype(int)

            # Each beat consists of a "lub" and a "dub"
            lub_duration = HeartbeatConstants.LUB_DURATION_SECONDS
            dub_duration = HeartbeatConstants.DUB_DURATION_SECONDS
            dub_delay = HeartbeatConstants.DUB_DELAY_SECONDS

            # Add slight natural variation to heartbeat amplitude
            if HAS_PERLIN and self.use_perlin:
                # Generate very slow perlin noise for natural amplitude variations
                amp_variation = generate_perlin_noise(
                    self.sample_rate, 
                    duration_seconds / 5, 
                    octaves=1, 
                    persistence=0.5,
                    seed=self.random_state.seed
                )
                
                # Stretch to get only a few variations over the whole duration
                indices = np.linspace(0, len(amp_variation) - 1, len(beat_samples))
                indices = np.clip(indices.astype(int), 0, len(amp_variation) - 1)
                amp_variations = 0.85 + HeartbeatConstants.AMPLITUDE_VARIATION_PCT * amp_variation[indices]
            else:
                # Fallback to random variations
                amp_variations = 0.85 + HeartbeatConstants.AMPLITUDE_VARIATION_PCT * self.random_state.normal(0, 0.3, len(beat_samples))
                
                # Smooth the variations
                amp_variations = np.convolve(amp_variations, np.ones(5) / 5, mode="same")

            # Create envelopes for lub and dub with slight variations
            for i, beat_start_sample in enumerate(beat_samples):
                if beat_start_sample >= samples:
                    continue

                # Convert to time for easier calculation
                beat_start = beat_start_sample / self.sample_rate

                # Amplitude variation for this beat
                amp_factor = amp_variations[min(i, len(amp_variations) - 1)]

                # "Lub" part
                lub_env_width = lub_duration / 5
                lub_samples = int(lub_duration * self.sample_rate)
                lub_end = min(beat_start_sample + lub_samples, samples)

                # Create time vector for this segment
                t_lub = (np.linspace(0, lub_end - beat_start_sample, lub_end - beat_start_sample) / 
                        self.sample_rate)

                # Create envelope and waveform
                lub_envelope = np.exp(-(t_lub**2) / (2 * lub_env_width**2))
                lub = lub_envelope * np.sin(2 * np.pi * HeartbeatConstants.LUB_FREQUENCY_HZ * t_lub) * amp_factor

                # Add to heartbeat sound
                heartbeat[beat_start_sample:lub_end] += lub

                # "Dub" part (softer and follows the lub)
                dub_start_sample = min(beat_start_sample + int(dub_delay * self.sample_rate), samples - 1)
                dub_env_width = dub_duration / 5
                dub_samples = int(dub_duration * self.sample_rate)
                dub_end = min(dub_start_sample + dub_samples, samples)

                # Check if there's room for the dub sound
                if dub_end > dub_start_sample:
                    # Create time vector for dub segment
                    t_dub = (np.linspace(0, dub_end - dub_start_sample, dub_end - dub_start_sample) / 
                            self.sample_rate)

                    # Create envelope and waveform (softer than lub)
                    dub_envelope = np.exp(-(t_dub**2) / (2 * dub_env_width**2))
                    dub = 0.7 * dub_envelope * np.sin(2 * np.pi * HeartbeatConstants.DUB_FREQUENCY_HZ * t_dub) * amp_factor

                    # Add to heartbeat sound
                    heartbeat[dub_start_sample:dub_end] += dub

            # Apply a subtle low-pass filter for more natural sound
            b, a = signal.butter(4, 200 / (self.sample_rate / 2), "low")
            heartbeat = signal.lfilter(b, a, heartbeat)

            # Normalize
            max_val = np.max(np.abs(heartbeat))
            if max_val > 0:
                heartbeat = heartbeat / max_val * HeartbeatConstants.DEFAULT_AMPLITUDE

            # Sanitize to ensure valid audio
            return self.sanitize_audio(heartbeat, max_amplitude=HeartbeatConstants.DEFAULT_AMPLITUDE)
            
        except Exception as e:
            logger.error(f"Error generating heartbeat: {e}")
            # Generate a simple fallback heartbeat in case of error
            return self._generate_simple_fallback_heartbeat(duration_seconds)
            
    def _generate_simple_fallback_heartbeat(self, duration_seconds: int) -> np.ndarray:
        """Generate simple heartbeat as fallback in case of error."""
        try:
            samples = int(duration_seconds * self.sample_rate)
            heartbeat = np.zeros(samples)
            
            # Simple constant rate heartbeat at 70 BPM
            bpm = 70
            period = 60 / bpm  # seconds per beat
            beat_interval_samples = int(period * self.sample_rate)
            
            for i in range(0, samples, beat_interval_samples):
                if i + 200 < samples:  # 200 samples for lub
                    heartbeat[i:i+200] = 0.7 * np.sin(2 * np.pi * 60 * np.linspace(0, 0.1, 200))
                if i + 500 < samples and i + 700 < samples:  # dub after 0.25s
                    heartbeat[i+500:i+700] = 0.4 * np.sin(2 * np.pi * 45 * np.linspace(0, 0.1, 200))
            
            return self.sanitize_audio(heartbeat, max_amplitude=0.7)
                
        except Exception as e:
            logger.error(f"Error generating fallback heartbeat: {e}")
            # Return simple pulse waveform as last resort
            samples = int(duration_seconds * self.sample_rate)
            pulse_rate = 1.2  # beats per second
            return 0.5 * np.sin(2 * np.pi * pulse_rate * np.linspace(0, duration_seconds, samples))

    def generate_shushing_sound(self, duration_seconds: int) -> np.ndarray:
        """
        Generate a more natural shushing sound similar to what parents do to calm babies
        
        Args:
            duration_seconds: Length of the sound in seconds
            
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

            # Create a more natural, human-like rhythm
            t = np.linspace(0, duration_seconds, samples, endpoint=False)

            if HAS_PERLIN and self.use_perlin:
                # Use perlin noise for organic, human-like variations
                shush_rhythm = generate_perlin_noise(
                    self.sample_rate,
                    duration_seconds, 
                    octaves=2, 
                    persistence=0.6,
                    seed=self.random_state.seed
                )

                # Scale the noise to the appropriate rhythm rate
                rhythm_scale = ShushingConstants.SHUSH_RATE_PER_SECOND
                indices = np.linspace(0, len(shush_rhythm) // 20, samples)
                indices = np.clip(indices.astype(int), 0, len(shush_rhythm) - 1)

                # Transform to min-max range for a more natural shushing pattern
                shush_modulation = (
                    ShushingConstants.MODULATION_MIN + 
                    (ShushingConstants.MODULATION_MAX - ShushingConstants.MODULATION_MIN) * 
                    (0.5 + 0.5 * shush_rhythm[indices])
                )
            else:
                # Fallback to sine-based rhythm with some randomness
                shush_rate = ShushingConstants.SHUSH_RATE_PER_SECOND
                phase_variation = 0.2 * self.random_state.normal(0, 1, int(duration_seconds * shush_rate) + 1)

                # Create a modified sine wave with phase variations
                shush_modulation = np.zeros(samples)
                for i in range(int(duration_seconds * shush_rate)):
                    start_idx = int(i * self.sample_rate / shush_rate)
                    end_idx = int((i + 1) * self.sample_rate / shush_rate)
                    end_idx = min(end_idx, samples)

                    # Create a single shush cycle with natural attack and decay
                    cycle_len = end_idx - start_idx
                    if cycle_len > 0:
                        cycle = (
                            ShushingConstants.MODULATION_MIN + 
                            (ShushingConstants.MODULATION_MAX - ShushingConstants.MODULATION_MIN) * 
                            np.sin(np.linspace(0 + phase_variation[i], np.pi + phase_variation[i], cycle_len)) ** 2
                        )
                        shush_modulation[start_idx:end_idx] = cycle

            # Apply the modulation
            shushing = shushing * shush_modulation

            # Apply subtle formant filtering to make it sound more like human shushing
            formant_filter = signal.firwin2(
                101,
                [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                [0.1, 0.2, 0.5, 1.0, 0.8, 0.6, 0.7, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1, 0.05],
                fs=self.sample_rate,
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

    def generate_fan_sound(self, duration_seconds: int) -> np.ndarray:
        """
        Generate a more realistic fan or air conditioner type sound
        
        Args:
            duration_seconds: Length of the sound in seconds
            
        Returns:
            Fan sound array
        """
        try:
            samples = int(duration_seconds * self.sample_rate)

            # Base noise is a mix of pink and white
            # Start with white noise
            white = self.random_state.normal(0, 0.5, samples)
            
            # Create pink noise using simple filtering
            pink_filter = signal.firwin(1001, 1000 / (self.sample_rate / 2), window='hann')
            pink = signal.lfilter(pink_filter, 1, white)
            pink = pink / np.max(np.abs(pink)) * 0.5  # Normalize
            
            # Mix white and pink noise
            mixed = white * 0.7 + pink * 0.3

            # Add a more natural, slightly irregular fan blade rotation
            t = np.linspace(0, duration_seconds, samples, endpoint=False)

            if HAS_PERLIN and self.use_perlin:
                # Use perlin noise for natural speed variations
                speed_variation = generate_perlin_noise(
                    self.sample_rate,
                    duration_seconds, 
                    octaves=1, 
                    persistence=0.5,
                    seed=self.random_state.seed
                )
                
                # Stretch to get only a few variations over the duration
                indices = np.linspace(0, len(speed_variation) // 100, samples)
                indices = np.clip(indices.astype(int), 0, len(speed_variation) - 1)
                
                # Average rotation speed with variations
                base_rotation = FanConstants.BASE_ROTATION_HZ
                rotation_speed = base_rotation * (1 + FanConstants.SPEED_VARIATION_PCT * speed_variation[indices])

                # Integrate speed to get phase
                phase = np.cumsum(rotation_speed) / self.sample_rate * 2 * np.pi

                # Calculate modulation with harmonic overtones (real fans have multiple harmonics)
                rotation_modulation = (0.08 * np.sin(phase) + 
                                      0.04 * np.sin(2 * phase) + 
                                      0.02 * np.sin(3 * phase))
            else:
                # Fallback to simpler fan simulation
                rotation_rate = FanConstants.BASE_ROTATION_HZ
                
                # Add a slight drift to the rotation speed
                drift = FanConstants.SPEED_VARIATION_PCT * np.sin(2 * np.pi * 0.05 * t)
                rotation_modulation = 0.1 * np.sin(2 * np.pi * rotation_rate * (1 + drift) * t)
                
                # Add some harmonics
                rotation_modulation += 0.05 * np.sin(2 * 2 * np.pi * rotation_rate * (1 + drift) * t)

            # Apply fan modulation
            fan_sound = mixed * (1 + rotation_modulation)

            # Apply resonance filtering to simulate fan housing
            # Fans often have specific resonant frequencies
            for freq in FanConstants.RESONANCE_FREQUENCIES_HZ:
                q_factor = FanConstants.RESONANCE_Q_FACTOR
                b, a = signal.iirpeak(freq, q_factor, self.sample_rate)
                fan_sound = signal.lfilter(b, a, fan_sound)

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
            # Extract parameters
            base_level_db = params.base_level_db
            cry_response_db = params.cry_response_db
            response_time_ms = params.response_time_ms

            # Convert dB difference to amplitude ratio
            level_increase = 10 ** (cry_response_db / 20.0)

            # Response time in samples
            response_samples = int(response_time_ms * self.sample_rate / 1000)

            # Create simulated cry periods (for demonstration)
            samples = len(audio)
            is_stereo = len(audio.shape) > 1

            # Simulate cry periods approximately every 5-10 minutes for 30-60 seconds
            cry_periods = []

            # At least 5 minutes between potential cry periods
            interval_min_samples = 5 * 60 * self.sample_rate

            # Create Perlin noise as a base for non-regular cry patterns
            if HAS_PERLIN and self.use_perlin:
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

                # Threshold for cry event
                cry_threshold = 0.6  # Adjust as needed

                # Find potential cry starts
                crossing_points = np.where(
                    np.diff((cry_likelihood > cry_threshold).astype(int)) > 0
                )[0]

                # Filter to ensure minimum spacing
                if len(crossing_points) > 0:
                    filtered_points = [crossing_points[0]]
                    for point in crossing_points[1:]:
                        if point - filtered_points[-1] >= interval_min_samples:
                            filtered_points.append(point)

                    # Create cry periods of random lengths between 30-60 seconds
                    for start in filtered_points:
                        if start < samples - interval_min_samples:
                            duration = self.random_state.randint(30, 60) * self.sample_rate
                            end = min(start + duration, samples)
                            cry_periods.append((start, end))
            else:
                # Fallback to regular intervals if no Perlin noise
                interval_samples = 8 * 60 * self.sample_rate  # 8 minutes
                for start in range(0, samples, interval_samples):
                    if start < samples - interval_min_samples:
                        duration = self.random_state.randint(30, 60) * self.sample_rate
                        end = min(start + duration, samples)
                        cry_periods.append((start, end))

            # Create envelope for shushing (base level, with increases during cry periods)
            envelope = np.ones(samples)

            # Apply response envelope for each cry period
            for start, end in cry_periods:
                # Gradual ramp up at start of cry
                ramp_up_end = min(start + response_samples, samples)
                ramp_up = np.linspace(1.0, level_increase, ramp_up_end - start)
                envelope[start:ramp_up_end] = ramp_up

                # Sustained increased level during cry
                sustain_end = max(ramp_up_end, min(end - response_samples, samples))
                envelope[ramp_up_end:sustain_end] = level_increase

                # Gradual ramp down at end of cry
                if sustain_end < samples:
                    ramp_down_end = min(sustain_end + response_samples, samples)
                    ramp_down = np.linspace(level_increase, 1.0, ramp_down_end - sustain_end)
                    envelope[sustain_end:ramp_down_end] = ramp_down

            # Apply envelope to shushing efficiently
            dynamic_shushing = apply_modulation(shushing, envelope)

            # Mix with original audio (assuming shushing is already mixed at 0.3 ratio)
            if is_stereo:
                output = audio * 0.7 + dynamic_shushing * 0.3
            else:
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
                    output = audio * 0.7 + shushing * 0.3
                else:
                    output = audio * 0.7 + shushing * 0.3
                    
                return self.sanitize_audio(output)
            except:
                # Return original audio if mixing fails
                return audio

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
            # Extract parameters
            mix_level_db = params.mix_level_db

            # Convert dB to linear scale
            mix_ratio = 10 ** (mix_level_db / 20.0)

            # Create simulated parental humming
            samples = len(audio)
            is_stereo = len(audio.shape) > 1

            # Basic parameters for a natural-sounding hum
            hum_freq = 220.0  # Typical female hum, ~A3
            hum_duration_seconds = 2.0
            hum_interval_seconds = 8.0  # Space between hums

            # Convert to samples
            hum_samples = int(hum_duration_seconds * self.sample_rate)
            interval_samples = int(hum_interval_seconds * self.sample_rate)

            # Create output array
            if is_stereo:
                hum_output = np.zeros((samples, audio.shape[1]))
            else:
                hum_output = np.zeros(samples)

            # Time vector for a single hum
            t_hum = np.linspace(0, hum_duration_seconds, hum_samples, endpoint=False)

            # Generate basic hum with harmonics for realism
            hum_base = np.sin(2 * np.pi * hum_freq * t_hum)
            hum_harmonic1 = 0.5 * np.sin(2 * np.pi * hum_freq * 2 * t_hum)  # First harmonic
            hum_harmonic2 = 0.3 * np.sin(2 * np.pi * hum_freq * 3 * t_hum)  # Second harmonic
            hum_harmonic3 = 0.1 * np.sin(2 * np.pi * hum_freq * 4 * t_hum)  # Third harmonic

            basic_hum = hum_base + hum_harmonic1 + hum_harmonic2 + hum_harmonic3

            # Apply envelope for natural attack and decay
            envelope = np.zeros_like(basic_hum)
            attack_samples = int(0.2 * hum_samples)  # 20% attack
            sustain_samples = int(0.5 * hum_samples)  # 50% sustain
            decay_samples = hum_samples - attack_samples - sustain_samples  # 30% decay

            # Create segments
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
            envelope[attack_samples : attack_samples + sustain_samples] = 1.0
            envelope[attack_samples + sustain_samples :] = np.linspace(1, 0, decay_samples)

            # Apply envelope
            hum = basic_hum * envelope

            # Apply vocal formant filtering for more realism
            # Simplified formant filter
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
                        for c in range(audio.shape[1]):
                            hum_output[start_idx:end_idx, c] = hum[:hum_len]
                    else:
                        hum_output[start_idx:end_idx] = hum[:hum_len]

            # Mix with original audio at specified level
            output = audio * (1 - mix_ratio) + hum_output * mix_ratio

            return self.sanitize_audio(output)

        except Exception as e:
            logger.error(f"Error applying parental voice: {e}")
            # Return original audio if voice processing fails
            return audio