"""
Natural sound generators like heartbeat, shushing, and fan sounds.
"""

import random
import numpy as np
from scipy import signal
import logging

from sound_profiles.base import SoundProfileGenerator
from models.parameters import HeartbeatParameters, DynamicShushing, ParentalVoice
from utils.optional_imports import HAS_PERLIN

logger = logging.getLogger("BabySleepSoundGenerator")

# Import optional libraries
if HAS_PERLIN:
    import noise


class NaturalSoundGenerator(SoundProfileGenerator):
    """Generator for natural sounds like heartbeats, shushing, and fan sounds."""
    
    def __init__(self, sample_rate: int, use_perlin: bool = True):
        """
        Initialize the natural sound generator.
        
        Args:
            sample_rate: Audio sample rate
            use_perlin: Whether to use Perlin noise for more natural variations
        """
        super().__init__(sample_rate, use_perlin)
    
    def generate_heartbeat(
        self,
        duration_seconds: int,
        heartbeat_params: HeartbeatParameters = None,
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
                bpm_noise = self._generate_perlin_noise(duration_seconds / 10, octaves=1, persistence=0.5)
                
                # Stretch the noise to match desired cycle period
                indices = np.linspace(0, len(bpm_noise) - 1, int(cycles * 100) + 1)
                indices = np.clip(indices.astype(int), 0, len(bpm_noise) - 1)
                bpm_variations = bpm_noise[indices]
                
                # Interpolate to full length
                full_indices = np.linspace(0, len(bpm_variations) - 1, int(duration_seconds) + 1)
                bpm_variations = np.interp(full_indices, np.arange(len(bpm_variations)), bpm_variations)
                
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
                # Get BPM at this time point
                current_bpm = np.interp(current_time, t, bpm_variations)
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
        lub_duration = 0.12  # in seconds
        dub_duration = 0.1  # in seconds
        dub_delay = 0.2  # time after lub

        # Add slight natural variation to heartbeat amplitude
        if HAS_PERLIN and self.use_perlin:
            # Generate very slow perlin noise for natural amplitude variations
            amp_variation = self._generate_perlin_noise(duration_seconds / 5, octaves=1, persistence=0.5)
            
            # Stretch to get only a few variations over the whole duration
            indices = np.linspace(0, len(amp_variation) - 1, len(beat_samples))
            indices = np.clip(indices.astype(int), 0, len(amp_variation) - 1)
            amp_variations = 0.85 + 0.15 * amp_variation[indices]  # 15% variation
        else:
            # Fallback to random variations
            amp_variations = 0.85 + 0.15 * np.random.normal(0, 0.3, len(beat_samples))
            
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
            lub = lub_envelope * np.sin(2 * np.pi * 60 * t_lub) * amp_factor

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
                dub = 0.7 * dub_envelope * np.sin(2 * np.pi * 45 * t_dub) * amp_factor

                # Add to heartbeat sound
                heartbeat[dub_start_sample:dub_end] += dub

        # Apply a subtle low-pass filter for more natural sound
        b, a = signal.butter(4, 200 / (self.sample_rate / 2), "low")
        heartbeat = signal.lfilter(b, a, heartbeat)

        # Normalize
        max_val = np.max(np.abs(heartbeat))
        if max_val > 0:
            heartbeat = heartbeat / max_val * 0.8

        return heartbeat

    def generate_shushing_sound(self, duration_seconds: int) -> np.ndarray:
        """
        Generate a more natural shushing sound similar to what parents do to calm babies
        
        Args:
            duration_seconds: Length of the sound in seconds
            
        Returns:
            Shushing sound array
        """
        samples = int(duration_seconds * self.sample_rate)

        # Start with filtered white noise
        white_noise = np.random.normal(0, 0.5, samples)

        # Apply bandpass filter to focus on "shh" frequencies (2000-4000 Hz)
        b, a = signal.butter(4, [2000 / (self.sample_rate / 2), 4000 / (self.sample_rate / 2)], "band")
        shushing = signal.lfilter(b, a, white_noise)

        # Create a more natural, human-like rhythm
        t = np.linspace(0, duration_seconds, samples, endpoint=False)

        if HAS_PERLIN and self.use_perlin:
            # Use perlin noise for organic, human-like variations
            shush_rhythm = self._generate_perlin_noise(duration_seconds, octaves=2, persistence=0.6)

            # Scale the noise to the appropriate rhythm rate
            rhythm_scale = 1.5  # Average shushes per second
            indices = np.linspace(0, len(shush_rhythm) // 20, samples)
            indices = np.clip(indices.astype(int), 0, len(shush_rhythm) - 1)

            # Transform to 0.3-1.0 range for a more natural shushing pattern
            shush_modulation = 0.3 + 0.7 * (0.5 + 0.5 * shush_rhythm[indices])
        else:
            # Fallback to sine-based rhythm with some randomness
            shush_rate = 1.5  # shushes per second
            phase_variation = 0.2 * np.random.normal(0, 1, int(duration_seconds * shush_rate) + 1)

            # Create a modified sine wave with phase variations
            shush_modulation = np.zeros(samples)
            for i in range(int(duration_seconds * shush_rate)):
                start_idx = int(i * self.sample_rate / shush_rate)
                end_idx = int((i + 1) * self.sample_rate / shush_rate)
                end_idx = min(end_idx, samples)

                # Create a single shush cycle with natural attack and decay
                cycle_len = end_idx - start_idx
                if cycle_len > 0:
                    cycle = (0.3 + 0.7 * np.sin(np.linspace(
                        0 + phase_variation[i],
                        np.pi + phase_variation[i],
                        cycle_len)) ** 2)
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

        # Normalize
        max_val = np.max(np.abs(shushing))
        if max_val > 0:
            shushing = shushing / max_val * 0.9

        return shushing

    def generate_fan_sound(self, duration_seconds: int) -> np.ndarray:
        """
        Generate a more realistic fan or air conditioner type sound
        
        Args:
            duration_seconds: Length of the sound in seconds
            
        Returns:
            Fan sound array
        """
        samples = int(duration_seconds * self.sample_rate)

        # Base noise is a mix of pink and white
        # Start with white noise
        white = np.random.normal(0, 0.5, samples)
        
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
            speed_variation = self._generate_perlin_noise(duration_seconds, octaves=1, persistence=0.5)
            
            # Stretch to get only a few variations over the duration
            indices = np.linspace(0, len(speed_variation) // 100, samples)
            indices = np.clip(indices.astype(int), 0, len(speed_variation) - 1)
            
            # Average rotation speed with variations
            base_rotation = 4.0  # Hz
            rotation_speed = base_rotation * (1 + 0.1 * speed_variation[indices])

            # Integrate speed to get phase
            phase = np.cumsum(rotation_speed) / self.sample_rate * 2 * np.pi

            # Calculate modulation with harmonic overtones (real fans have multiple harmonics)
            rotation_modulation = (0.08 * np.sin(phase) + 
                                  0.04 * np.sin(2 * phase) + 
                                  0.02 * np.sin(3 * phase))
        else:
            # Fallback to simpler fan simulation
            rotation_rate = 4.0  # rotations per second
            
            # Add a slight drift to the rotation speed
            drift = 0.05 * np.sin(2 * np.pi * 0.05 * t)  # 5% speed drift over 20 seconds
            rotation_modulation = 0.1 * np.sin(2 * np.pi * rotation_rate * (1 + drift) * t)
            
            # Add some harmonics
            rotation_modulation += 0.05 * np.sin(2 * 2 * np.pi * rotation_rate * (1 + drift) * t)

        # Apply fan modulation
        fan_sound = mixed * (1 + rotation_modulation)

        # Apply resonance filtering to simulate fan housing
        # Fans often have specific resonant frequencies
        resonance_freqs = [180, 320, 560, 820]  # Hz
        for freq in resonance_freqs:
            q_factor = 10.0  # Narrowness of resonance
            b, a = signal.iirpeak(freq, q_factor, self.sample_rate)
            fan_sound = signal.lfilter(b, a, fan_sound)

        # Apply final bandpass filter to shape the spectrum
        b, a = signal.butter(3, [80 / (self.sample_rate / 2), 4000 / (self.sample_rate / 2)], "band")
        fan_sound = signal.lfilter(b, a, fan_sound)

        # Normalize
        max_val = np.max(np.abs(fan_sound))
        if max_val > 0:
            fan_sound = fan_sound / max_val * 0.85

        return fan_sound

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

        # Extract parameters
        base_level_db = params.base_level_db
        cry_response_db = params.cry_response_db
        response_time_ms = params.response_time_ms

        # Convert dB difference to amplitude ratio
        level_increase = 10 ** (cry_response_db / 20.0)

        # Response time in samples
        response_samples = int(response_time_ms * self.sample_rate / 1000)

        # Create simulated cry periods (for demonstration)
        # In a real implementation, this would come from actual cry detection
        samples = len(audio)
        is_stereo = len(audio.shape) > 1

        # Simulate cry periods approximately every 5-10 minutes for 30-60 seconds
        cry_periods = []

        # At least 5 minutes between potential cry periods
        interval_min_samples = 5 * 60 * self.sample_rate

        # Create Perlin noise as a base for non-regular cry patterns
        if HAS_PERLIN and self.use_perlin:
            # Generate very slow perlin noise for cry likelihood
            cry_likelihood = self._generate_perlin_noise(
                samples / self.sample_rate / 10, octaves=1, persistence=0.5
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
                        duration = random.randint(30, 60) * self.sample_rate
                        end = min(start + duration, samples)
                        cry_periods.append((start, end))
        else:
            # Fallback to regular intervals if no Perlin noise
            interval_samples = 8 * 60 * self.sample_rate  # 8 minutes
            for start in range(0, samples, interval_samples):
                if start < samples - interval_min_samples:
                    duration = random.randint(30, 60) * self.sample_rate
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
        if is_stereo:
            dynamic_shushing = shushing * envelope[:, np.newaxis]
        else:
            dynamic_shushing = shushing * envelope

        # Mix with original audio (assuming shushing is already mixed at 0.3 ratio)
        if is_stereo:
            output = audio * 0.7 + dynamic_shushing * 0.3
        else:
            output = audio * 0.7 + dynamic_shushing * 0.3

        # Normalize if needed
        max_val = np.max(np.abs(output))
        if max_val > 0.95:  # Constants.MAX_AUDIO_VALUE
            output = output / max_val * 0.95

        return output

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
                    hum_output[start_idx:end_idx] = hum[:hum_len, np.newaxis]
                else:
                    hum_output[start_idx:end_idx] = hum[:hum_len]

        # Mix with original audio at specified level
        output = audio * (1 - mix_ratio) + hum_output * mix_ratio

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