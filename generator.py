"""
Core generator class that orchestrates sound generation and processing.
"""

import os
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from models.constants import (
    Constants, FrequencyFocus, RoomSize, LoopingMethod, OutputFormat,
    PerformanceConstants, HeartbeatConstants, ShushingConstants, FanConstants,
    WombConstants, UmbilicalConstants, SpatialConstants
)
from models.parameters import (
    HeartbeatParameters, FrequencyEmphasis, LowPassFilter, FrequencyLimiting, CircadianAlignment,
    DynamicVolume, MotionSmoothing, MoroReflexPrevention, SleepCycleModulation, 
    DynamicShushing, BreathingModulation, SafetyFeatures, ParentalVoice, ProblemProfile
)
from models.sound_config import SoundConfiguration

from sound_profiles.noise import NoiseGenerator
from sound_profiles.natural import NaturalSoundGenerator
from sound_profiles.womb import WombSoundGenerator
from sound_profiles.factory import SoundGeneratorFactory

from processing.spatial import SpatialProcessor
from processing.room import RoomAcousticsProcessor
from processing.frequency import FrequencyProcessor
from processing.modulation import ModulationProcessor

from effects.breathing import BreathingModulator
from effects.sleep_cycles import SleepCycleModulator
from effects.dynamic_volume import DynamicVolumeProcessor
from effects.reflex import ReflexPreventer

from output.normalization import LoudnessNormalizer
from output.export import AudioExporter
from output.visualization import SpectrumVisualizer

from utils.optional_imports import (
    HAS_PERLIN, HAS_LOUDNORM, HAS_LIBROSA, HAS_PYROOMACOUSTICS,
    get_matplotlib_plt, get_soundfile_module
)
from utils.random_state import RandomStateManager
from utils.config import ConfigManager
from utils.progress import ProgressReporter
from utils.parallel import ParallelProcessor

logger = logging.getLogger("BabySleepSoundGenerator")


class AudioProcessor:
    """
    Handles audio processing operations like fading, mixing, and looping.
    """
    
    def __init__(self, sample_rate: int, crossfade_duration: float = Constants.DEFAULT_CROSSFADE_DURATION):
        """
        Initialize the audio processor.
        
        Args:
            sample_rate: Audio sample rate
            crossfade_duration: Default crossfade duration in seconds
        """
        self.sample_rate = sample_rate
        self.crossfade_duration = crossfade_duration
    
    def add_fade(
        self,
        audio: np.ndarray,
        fade_in_seconds: float = 10.0,
        fade_out_seconds: float = 10.0,
    ) -> np.ndarray:
        """
        Add smooth, natural fade in and fade out to the audio.
        Uses an equal-power curve for more natural fades than linear.
        
        Args:
            audio: Input audio array
            fade_in_seconds: Duration of fade-in in seconds
            fade_out_seconds: Duration of fade-out in seconds
            
        Returns:
            Audio with fades applied
        """
        # Validate input
        if fade_in_seconds < 0 or fade_out_seconds < 0:
            raise ValueError("Fade durations must be non-negative")
        
        try:
            fade_in_samples = int(fade_in_seconds * self.sample_rate)
            fade_out_samples = int(fade_out_seconds * self.sample_rate)
            
            # Check if audio is long enough for the requested fades
            total_samples = len(audio)
            if fade_in_samples + fade_out_samples > total_samples:
                logger.warning("Fade durations exceed audio length. Adjusting fade times.")
                ratio = total_samples / (fade_in_samples + fade_out_samples)
                fade_in_samples = int(fade_in_samples * ratio * 0.5)
                fade_out_samples = int(fade_out_samples * ratio * 0.5)

            # Check for stereo or mono
            is_stereo = len(audio.shape) > 1
            
            # Create a copy of the input audio to avoid modifying the original
            result = audio.copy()
            
            # Create the fades using a sine curve instead of linear for more natural transition
            fade_in = np.sin(np.linspace(0, np.pi / 2, fade_in_samples)) ** 2
            fade_out = np.sin(np.linspace(np.pi / 2, 0, fade_out_samples)) ** 2

            # Apply fades efficiently using broadcasting
            if is_stereo:
                # Stereo
                if fade_in_samples > 0:
                    result[:fade_in_samples] *= fade_in[:, np.newaxis]
                if fade_out_samples > 0:
                    result[-fade_out_samples:] *= fade_out[:, np.newaxis]
            else:
                # Mono
                if fade_in_samples > 0:
                    result[:fade_in_samples] *= fade_in
                if fade_out_samples > 0:
                    result[-fade_out_samples:] *= fade_out

            return result
            
        except Exception as e:
            logger.error(f"Error applying fade: {e}")
            # Return original audio if fading fails
            return audio

    def mix_sounds(
        self,
        primary: np.ndarray,
        overlays: List[np.ndarray],
        mix_ratios: List[float],
        motion_smoothing: Optional[MotionSmoothing] = None,
    ) -> np.ndarray:
        """
        Mix multiple sound arrays together with specified mix ratios.
        Uses in-place operations where possible to reduce memory usage.
        """
        if not overlays or not mix_ratios:
            return primary

        # Validate input
        if len(overlays) != len(mix_ratios):
            raise ValueError("Number of overlays must match number of mix ratios")
            
        # Check if primary is stereo
        is_stereo = len(primary.shape) > 1

        # Create a copy only once at the beginning
        mixed = primary.copy()

        # Pre-allocate a stereo buffer for mono-to-stereo conversion if needed
        stereo_buffer = None
        if is_stereo:
            channels = primary.shape[1]
            stereo_buffer = np.zeros((0, channels))  # Empty buffer, will resize as needed

        # Add each overlay with its mix ratio
        for i, (overlay, ratio) in enumerate(zip(overlays, mix_ratios)):
            # Skip invalid overlays
            if overlay is None or overlay.size == 0:
                logger.warning(f"Skipping empty overlay at index {i}")
                continue
                
            # Handle shape conversion properly
            if len(overlay.shape) == 1 and is_stereo:
                # Convert mono overlay to stereo using broadcasting
                try:
                    # Resize buffer if needed
                    if len(overlay) > len(stereo_buffer):
                        stereo_buffer = np.zeros((len(overlay), channels))
                    
                    # Use buffer up to required length
                    buffer_view = stereo_buffer[:len(overlay)]
                    
                    # Use broadcasting for efficient assignment
                    buffer_view[:] = overlay[:, np.newaxis]
                    overlay_stereo = buffer_view
                    
                    # Mix in-place
                    np.multiply(mixed, (1 - ratio), out=mixed)
                    np.add(mixed, overlay_stereo * ratio, out=mixed)
                except Exception as e:
                    logger.error(f"Error converting overlay to stereo: {e}")
                    continue
            elif is_stereo and len(overlay.shape) > 1:
                # Both are stereo, mix in-place
                try:
                    # Ensure overlay is the right length
                    if len(overlay) > len(mixed):
                        overlay = overlay[:len(mixed)]
                    elif len(overlay) < len(mixed):
                        # Skip if too short - would need padding
                        logger.warning(f"Skipping overlay {i} - too short and would need padding")
                        continue
                    
                    # Mix in-place
                    np.multiply(mixed, (1 - ratio), out=mixed)
                    np.add(mixed, overlay * ratio, out=mixed)
                except Exception as e:
                    logger.error(f"Error mixing stereo overlay {i}: {e}")
                    continue
            else:
                # Both are mono or mixed case
                try:
                    # Ensure overlay is the right length
                    if len(overlay) > len(mixed):
                        overlay = overlay[:len(mixed)]
                    elif len(overlay) < len(mixed):
                        # Skip if too short - would need padding
                        logger.warning(f"Skipping overlay {i} - too short and would need padding")
                        continue
                    
                    # Mix in-place
                    if is_stereo:
                        for c in range(mixed.shape[1]):
                            np.multiply(mixed[:, c], (1 - ratio), out=mixed[:, c])
                            np.add(mixed[:, c], overlay * ratio, out=mixed[:, c])
                    else:
                        np.multiply(mixed, (1 - ratio), out=mixed)
                        np.add(mixed, overlay * ratio, out=mixed)
                except Exception as e:
                    logger.error(f"Error mixing overlay {i}: {e}")
                    continue

        # Normalize to prevent clipping (in-place)
        max_val = np.max(np.abs(mixed))
        if max_val > Constants.MAX_AUDIO_VALUE:
            np.multiply(mixed, Constants.MAX_AUDIO_VALUE / max_val, out=mixed)

        return mixed
        
    def find_best_loop_point(
        self, audio: np.ndarray, segment_duration: float = 60.0
    ) -> int:
        """
        Find the best loop point using cross-correlation.
        This creates much more seamless loops by finding naturally matching points.

        Args:
            audio: The audio array
            segment_duration: Duration in seconds to use for matching

        Returns:
            Index of best loop point
        """
        # Validate input
        if len(audio) <= 0:
            logger.warning("Empty audio provided to find_best_loop_point")
            return 0
            
        # Don't try to analyze too much data - use segments
        segment_samples = int(segment_duration * self.sample_rate)
        
        # We'll look for matches between beginning and end segments
        if len(audio) <= 2 * segment_samples:
            # If audio is too short, adjust segment size
            segment_samples = len(audio) // 4
            if segment_samples <= 0:
                logger.warning("Audio too short for loop point detection")
                return len(audio) // 2

        # For stereo, analyze first channel only
        if len(audio.shape) > 1:
            analysis_channel = audio[:, 0]
        else:
            analysis_channel = audio

        # Get beginning segment
        begin_segment = analysis_channel[:segment_samples]

        # Pre-compute the FFT of the beginning segment once for efficiency
        begin_segment_fft = np.fft.rfft(begin_segment)
        begin_segment_conj = np.conjugate(begin_segment_fft)

        # Try different positions near the end to find best correlation
        best_correlation = -np.inf
        best_position = len(analysis_channel) - segment_samples

        # Search the last portion of the audio for good matching points
        search_start = int(len(analysis_channel) * Constants.LOOP_SEARCH_START_PERCENTAGE)

        # To make search faster, we'll check candidates at regular intervals
        step = self.sample_rate // Constants.LOOP_SEARCH_STEP_DIVIDER  # Check every fraction of a second

        # First pass: coarse search
        correlations = []
        positions = []
        
        try:
            for pos in range(search_start, len(analysis_channel) - segment_samples, step):
                positions.append(pos)
                end_segment = analysis_channel[pos : pos + segment_samples]
                
                # Use FFT-based correlation for all segments (more efficient)
                # Compute FFT of end segment
                end_segment_fft = np.fft.rfft(end_segment)
                
                # Compute correlation in frequency domain
                correlation = np.fft.irfft(end_segment_fft * begin_segment_conj)
                max_corr = np.max(np.abs(correlation))
                correlations.append(max_corr)
                
        except Exception as e:
            logger.error(f"Error during coarse loop point search: {e}")
            # Fallback to a reasonable default
            return len(analysis_channel) - segment_samples

        # Find the best match from the coarse search
        if not correlations:
            logger.warning("No correlation points found, using default loop point")
            return len(analysis_channel) - segment_samples
            
        try:    
            best_idx = np.argmax(correlations)
            best_correlation = correlations[best_idx]
            best_position = positions[best_idx]

            # Second pass: fine-tune by checking neighboring points
            fine_range = self.sample_rate // 2  # Check +/- 500ms around best point
            for pos in range(
                max(search_start, best_position - fine_range),
                min(len(analysis_channel) - segment_samples, best_position + fine_range),
                self.sample_rate // 20  # Check every 50ms for fine tuning
            ):
                end_segment = analysis_channel[pos : pos + segment_samples]
                
                # Use FFT-based correlation for fine tuning too
                end_segment_fft = np.fft.rfft(end_segment)
                correlation = np.fft.irfft(end_segment_fft * begin_segment_conj)
                max_corr = np.max(np.abs(correlation))

                if max_corr > best_correlation:
                    best_correlation = max_corr
                    best_position = pos
        except Exception as e:
            logger.error(f"Error during fine loop point search: {e}")
            # If fine-tuning fails, keep the result from coarse search
            # (or the default if that also failed)

        return best_position

    def create_seamless_loop(
        self, audio: np.ndarray, crossfade_seconds: Optional[float] = None
    ) -> np.ndarray:
        """
        Create a seamless loop with intelligent crossfade between end and beginning.
        Uses cross-correlation to find the best loop points.
        
        Args:
            audio: Input audio to loop
            crossfade_seconds: Duration of crossfade in seconds (uses default if None)
            
        Returns:
            Audio with seamless loop applied
        """
        if crossfade_seconds is None:
            crossfade_seconds = self.crossfade_duration
        
        try:    
            # Find the best place to create the loop
            loop_point = self.find_best_loop_point(audio)
            # Use this as the end of our looped segment
            audio = audio[:loop_point]

            crossfade_samples = int(crossfade_seconds * self.sample_rate)

            # Make sure crossfade isn't longer than half the audio
            if crossfade_samples > len(audio) // 2:
                crossfade_samples = len(audio) // 2

            # Handle stereo or mono
            is_stereo = len(audio.shape) > 1

            # Extract the beginning and end portions for crossfade
            if is_stereo:
                beginning = audio[:crossfade_samples].copy()
                end = audio[-crossfade_samples:].copy()
            else:
                beginning = audio[:crossfade_samples].copy()
                end = audio[-crossfade_samples:].copy()

            # Create crossfade weights using equal power crossfade for smoothest transition
            # This sounds more natural than linear crossfade
            fade_out = np.cos(np.linspace(0, np.pi / 2, crossfade_samples)) ** 2
            fade_in = np.sin(np.linspace(0, np.pi / 2, crossfade_samples)) ** 2

            # Apply crossfade efficiently
            if is_stereo:
                # Create crossfade for stereo
                crossfaded = end * fade_out[:, np.newaxis] + beginning * fade_in[:, np.newaxis]
                
                # Replace the end with the crossfaded section
                looped = audio.copy()
                looped[-crossfade_samples:] = crossfaded
            else:
                # Create crossfade for mono
                crossfaded = end * fade_out + beginning * fade_in
                
                # Replace the end with the crossfaded section
                looped = audio.copy()
                looped[-crossfade_samples:] = crossfaded

            return looped
            
        except Exception as e:
            logger.error(f"Error creating seamless loop: {e}")
            # Return original audio if looping fails
            return audio


class OutputManager:
    """
    Manages output operations and safety information.
    """
    
    def __init__(self, volume_to_db_spl: Dict[float, float], safe_duration_hours: Dict[float, float]):
        """
        Initialize the output manager.
        
        Args:
            volume_to_db_spl: Dictionary mapping normalized volumes to dB SPL values
            safe_duration_hours: Dictionary mapping dB SPL levels to safe listening hours
        """
        self.volume_to_db_spl = volume_to_db_spl
        self.safe_duration_hours = safe_duration_hours
    
    def print_safety_information(self, volume: float) -> None:
        """Print safety information based on the volume level"""
        try:
            # Find the closest SPL value for the given volume
            closest_vol = min(self.volume_to_db_spl.keys(), key=lambda x: abs(x - volume))
            closest_spl = self.volume_to_db_spl[closest_vol]

            logger.info("\n----- SAFETY INFORMATION -----")
            logger.info(f"Estimated SPL at 1 meter: {closest_spl:.1f} dB")

            # Calculate SPL at 7 feet (standard crib distance per AAP)
            spl_at_7ft = closest_spl - 20 * np.log10(2.1)  # 7ft ≈ 2.1m
            logger.info(f"Estimated SPL at 7 feet (crib distance): {spl_at_7ft:.1f} dB")

            # Provide safety guidance
            if closest_spl > 70:
                logger.warning("WARNING: Volume exceeds AAP recommendations (max 65-70 dB).")
                logger.warning("Please reduce volume to protect infant hearing.")
            elif closest_spl > 65:
                logger.warning("NOTICE: Volume is at the upper limit of AAP recommendations.")
                logger.warning("Consider reducing for extended use.")
            else:
                logger.info("SAFE: Volume is within AAP recommended limits for infant use.")

            # Duration guidance
            for spl_threshold, safe_hours in sorted(self.safe_duration_hours.items()):
                if closest_spl >= spl_threshold:
                    logger.info(f"Maximum recommended exposure: {safe_hours:.1f} hours at this volume level")
                    break

            logger.info("-------------------------------")
            
        except Exception as e:
            logger.error(f"Error displaying safety information: {e}")


class ProfileManager:
    """
    Manages problem profiles for different baby sleep issues.
    """
    
    def __init__(self):
        """Initialize the profile manager."""
        self.problem_profiles = {}
        self._initialize_problem_profiles()
    
    def _initialize_problem_profiles(self) -> None:
        """Initialize problem profiles using dataclasses for better structure"""
        from models.profiles import create_problem_profiles
        try:
            self.problem_profiles = create_problem_profiles()
        except Exception as e:
            logger.error(f"Error initializing problem profiles: {e}")
            # Create empty profiles as fallback
            self.problem_profiles = {}
    
    def get_profile(self, problem: str) -> Optional[ProblemProfile]:
        """
        Get a specific problem profile.
        
        Args:
            problem: Name of the problem profile
            
        Returns:
            Problem profile or None if not found
        """
        return self.problem_profiles.get(problem)
    
    def get_all_profile_names(self) -> List[str]:
        """
        Get all available profile names.
        
        Returns:
            List of profile names
        """
        return list(self.problem_profiles.keys())
    
    def get_profile_descriptions(self) -> Dict[str, str]:
        """
        Get descriptions for all profiles.
        
        Returns:
            Dictionary mapping profile names to descriptions
        """
        return {
            problem: profile.description
            for problem, profile in self.problem_profiles.items()
        }


class BabySleepSoundGenerator:
    """Generate broadcast-quality sleep sounds for babies based on developmental needs"""

    def __init__(
        self,
        sample_rate: int = Constants.DEFAULT_SAMPLE_RATE,
        bit_depth: int = Constants.DEFAULT_BIT_DEPTH,
        channels: int = Constants.DEFAULT_CHANNELS,
        target_loudness: float = Constants.DEFAULT_TARGET_LOUDNESS,
        use_hrtf: bool = True,
        room_simulation: bool = True,
        config_file: Optional[str] = None,
        seed: Optional[int] = None,
        use_equal_loudness: bool = True,
        use_limiter: bool = True,
        use_organic_drift: bool = True,
        use_diffusion: bool = False,
    ):
        """Initialize the generator with professional audio quality settings"""
        # Load configuration if provided
        if config_file:
            try:
                self.config = ConfigManager.get_instance(config_file)
                # Use values from config or fall back to defaults
                sample_rate = self.config.get_int("DEFAULT", "sample_rate", sample_rate)
                bit_depth = self.config.get_int("DEFAULT", "bit_depth", bit_depth)
                channels = self.config.get_int("DEFAULT", "channels", channels)
                target_loudness = self.config.get_float("DEFAULT", "target_loudness", target_loudness)
                use_hrtf = self.config.get_bool("DEFAULT", "use_hrtf", use_hrtf)
                room_simulation = self.config.get_bool("DEFAULT", "room_simulation", room_simulation)
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")
                logger.warning("Using default settings")
                self.config = None
        else:
            self.config = None
        
        # Initialize random state manager with the seed
        self.random_state = RandomStateManager.get_instance(seed)
        self.seed = self.random_state.seed  # Store the actual seed used
        
        self.sample_rate = sample_rate
        self.bit_depth = bit_depth  # 16, 24, or 32
        self.channels = channels
        self.target_loudness = target_loudness  # LUFS
        self.use_hrtf = use_hrtf and HAS_LIBROSA
        self.room_simulation = room_simulation and HAS_PYROOMACOUSTICS

        # Store new noise enhancement parameters
        self.use_equal_loudness = use_equal_loudness
        self.use_limiter = use_limiter
        self.use_organic_drift = use_organic_drift
        self.use_diffusion = use_diffusion

        # Set modulation parameters for natural variation
        self.use_dynamic_modulation = True
        self.modulation_depth = 0.08  # 8% variation
        self.modulation_rate = 0.002  # Very slow modulation (cycles per second)
        self.use_perlin = HAS_PERLIN  # Use Perlin noise if available

        # Choose appropriate looping method
        self.looping_method = LoopingMethod.CROSS_CORRELATION
        self.crossfade_duration = Constants.DEFAULT_CROSSFADE_DURATION

        # Reference volume calibration (mapping normalized values to dB SPL at 1m)
        # Based on clinical standards for infant sound therapy
        self.volume_to_db_spl = {
            0.3: 55.0,  # 55 dB SPL at 1m
            0.4: 58.0,  # 58 dB SPL at 1m
            0.5: 62.0,  # 62 dB SPL at 1m
            0.6: 66.0,  # 66 dB SPL at 1m (AAP max guideline)
            0.7: 70.0,  # 70 dB SPL at 1m (exceeds AAP guidelines)
            0.8: 75.0,  # 75 dB SPL at 1m (well above recommendations)
            0.9: 80.0,  # 80 dB SPL at 1m (excessive)
        }

        # Safe listening duration at various levels (based on WHO guidelines)
        self.safe_duration_hours = Constants.WHO_SAFE_HOURS

        # Initialize component generators using the factory pattern
        try:
            self.noise_generator = SoundGeneratorFactory.create_generator(
                "noise",
                sample_rate,
                self.use_perlin,
                modulation_depth=self.modulation_depth,
                seed=self.seed,
                use_equal_loudness=self.use_equal_loudness,
                use_limiter=self.use_limiter,
                use_organic_drift=self.use_organic_drift,
                use_diffusion=self.use_diffusion
            )
            
            self.natural_generator = SoundGeneratorFactory.create_generator(
                "natural",
                sample_rate,
                self.use_perlin,
                seed=self.seed
            )
            
            self.womb_generator = SoundGeneratorFactory.create_generator(
                "womb",
                sample_rate,
                self.use_perlin,
                seed=self.seed
            )
        except Exception as e:
            logger.error(f"Error initializing sound generators: {e}")
            raise

        # Initialize component classes
        try:
            # Create audio processor
            self.audio_processor = AudioProcessor(sample_rate, self.crossfade_duration)
            
            # Create output manager
            self.output_manager = OutputManager(self.volume_to_db_spl, self.safe_duration_hours)
            
            # Create profile manager
            self.profile_manager = ProfileManager()
            
            # Initialize processors
            self.spatial_processor = SpatialProcessor(sample_rate, use_hrtf)
            self.room_processor = RoomAcousticsProcessor(sample_rate, room_simulation)
            self.frequency_processor = FrequencyProcessor(sample_rate)
            self.modulation_processor = ModulationProcessor(
                sample_rate, self.use_perlin, depth=self.modulation_depth, rate=self.modulation_rate
            )
            
            # Initialize effect modules
            self.breathing_modulator = BreathingModulator(sample_rate, self.use_perlin)
            self.sleep_cycle_modulator = SleepCycleModulator(sample_rate, self.use_perlin)
            self.dynamic_volume_processor = DynamicVolumeProcessor(sample_rate, self.volume_to_db_spl)
            self.reflex_preventer = ReflexPreventer(sample_rate)
            
            # Initialize output modules
            self.loudness_normalizer = LoudnessNormalizer(sample_rate, target_loudness)
            self.audio_exporter = AudioExporter(sample_rate, bit_depth, channels)
            self.spectrum_visualizer = SpectrumVisualizer(sample_rate)
            
            # Initialize parallel processor
            self.parallel_processor = ParallelProcessor()
        except Exception as e:
            logger.error(f"Error initializing processors: {e}")
            raise
        
        # Define available sound profiles with updated factory pattern
        self.sound_profiles = {
            "white": lambda duration_seconds, **kwargs: 
                self.noise_generator.generate(duration_seconds, sound_type="white", **kwargs),
            "pink": lambda duration_seconds, **kwargs: 
                self.noise_generator.generate(duration_seconds, sound_type="pink", **kwargs),
            "brown": lambda duration_seconds, **kwargs: 
                self.noise_generator.generate(duration_seconds, sound_type="brown", **kwargs),
            "womb": lambda duration_seconds, **kwargs: 
                self.womb_generator.generate(duration_seconds, sound_type="womb", **kwargs),
            "heartbeat": lambda duration_seconds, **kwargs: 
                self.natural_generator.generate(duration_seconds, sound_type="heartbeat", **kwargs),
            "shushing": lambda duration_seconds, **kwargs: 
                self.natural_generator.generate(duration_seconds, sound_type="shushing", **kwargs),
            "fan": lambda duration_seconds, **kwargs: 
                self.natural_generator.generate(duration_seconds, sound_type="fan", **kwargs),
            "umbilical_swish": lambda duration_seconds, **kwargs: 
                self.womb_generator.generate(duration_seconds, sound_type="umbilical_swish", **kwargs),
        }

    def generate_from_config(self, config: SoundConfiguration, progress_callback=None) -> Optional[str]:
        """
        Generate a sound file based on a complete sound configuration.
        
        Args:
            config: Sound configuration
            progress_callback: Optional callback for progress updates
            
        Returns:
            Path to the generated audio file or None if generation failed
        """
        # Validate configuration
        if not config.validate():
            raise ValueError("Invalid sound configuration")
            
        # Set random seed if provided
        if config.seed is not None:
            self.random_state.set_seed(config.seed)
            
        # Create progress reporter
        total_steps = 10  # Approximate number of major steps
        progress = ProgressReporter(
            total_steps, 
            f"Generating {config.name}", 
            progress_callback
        )
        
        # Calculate duration in seconds
        duration_seconds = int(config.duration_seconds)
        
        # Set up tracking for any temp files or resources that need cleanup
        temp_files = []
        output_path = None
        
        try:
            # 1. Generate primary noise
            progress.update(0, force=True, status_info={"current_operation": "Initializing"})
            logger.info(f"Generating {config.primary_sound} sound...")
            progress.update(0, status_info={"current_operation": f"Generating {config.primary_sound}"})
            
            # Get the appropriate generator
            generator_type = SoundGeneratorFactory.get_generator_type_for_sound(config.primary_sound)
            generator = SoundGeneratorFactory.create_generator(
                generator_type, 
                config.sample_rate, 
                self.use_perlin, 
                config.seed
            )
            
            primary_sound = generator.generate(
                duration_seconds,
                sound_type=config.primary_sound,
                **config.primary_sound_params
            )
            progress.update(1)
            
            # 2. Apply frequency shaping
            logger.info(f"Applying {config.frequency_focus} frequency shaping...")
            progress.update(0, status_info={"current_operation": "Applying frequency shaping"})
            
            # Extract additional frequency parameters if present
            additional_params = {}
            if config.frequency_emphasis is not None:
                additional_params["frequency_emphasis"] = FrequencyEmphasis(**config.frequency_emphasis)
            if config.low_pass_filter is not None:
                additional_params["low_pass_filter"] = LowPassFilter(**config.low_pass_filter)
            
            shaped_noise = self.frequency_processor.apply_frequency_shaping(
                primary_sound, config.frequency_focus, additional_params
            )
            progress.update(1)
            
            # 3. Generate overlay sounds in parallel if needed
            overlay_sounds = []
            mix_ratios = []
            
            if config.overlay_sounds:
                logger.info("Generating overlay sounds...")
                progress.update(0, status_info={
                    "current_operation": f"Generating {len(config.overlay_sounds)} overlay sounds"
                })
                
                # Create tasks for parallel processing
                overlay_tasks = []
                
                for overlay_config in config.overlay_sounds:
                    sound_type = overlay_config.get("type")
                    mix_ratio = overlay_config.get("mix_ratio", 0.2)
                    
                    if not sound_type:
                        logger.warning(f"Skipping overlay with missing type: {overlay_config}")
                        continue
                    
                    # Use a copy of the config without the type and mix_ratio
                    overlay_params = overlay_config.copy()
                    if "type" in overlay_params:
                        overlay_params.pop("type")
                    if "mix_ratio" in overlay_params:
                        overlay_params.pop("mix_ratio")
                    
                    # Get the appropriate generator
                    generator_type = SoundGeneratorFactory.get_generator_type_for_sound(sound_type)
                    
                    # Create generator instance
                    overlay_generator = SoundGeneratorFactory.create_generator(
                        generator_type, 
                        config.sample_rate, 
                        self.use_perlin, 
                        config.seed
                    )
                    
                    # Add task for parallel processing
                    overlay_tasks.append((
                        overlay_generator.generate,
                        {
                            "duration_seconds": duration_seconds,
                            "sound_type": sound_type,
                            **overlay_params
                        }
                    ))
                    
                    # Store mix ratio
                    mix_ratios.append(mix_ratio)
                
                # Process in parallel if there are multiple overlays
                if len(overlay_tasks) > 1:
                    overlay_results = self.parallel_processor.process(
                        overlay_tasks, 
                        description="Generate overlay sounds"
                    )
                    overlay_sounds = [result for result in overlay_results if result is not None]
                elif len(overlay_tasks) == 1:
                    # Just process directly for a single overlay
                    func, kwargs = overlay_tasks[0]
                    overlay_sounds = [func(**kwargs)]
                
                # Mix sounds if we have overlays
                if overlay_sounds:
                    logger.info("Mixing sounds...")
                    progress.update(0, status_info={"current_operation": "Mixing sounds"})
                    
                    # Create motion smoothing parameters if specified
                    motion_smoothing = None
                    if hasattr(config, "motion_smoothing") and config.motion_smoothing:
                        motion_smoothing = MotionSmoothing(**config.motion_smoothing)
                    
                    shaped_noise = self.audio_processor.mix_sounds(
                        shaped_noise, overlay_sounds, mix_ratios, motion_smoothing
                    )
            
            progress.update(1)
            
            # 4. Apply spatial processing
            logger.info("Applying spatial processing...")
            progress.update(0, status_info={"current_operation": "Applying spatial processing"})
            
            # Apply HRTF-based spatialization or basic stereo widening
            if len(shaped_noise.shape) == 1:  # Only apply if mono
                stereo_audio = self.spatial_processor.apply_hrtf_spatialization(
                    shaped_noise, width=config.spatial_width
                )
            else:
                # Already stereo, just adjust width
                stereo_audio = self.spatial_processor.apply_basic_stereo_widening(
                    shaped_noise, width=config.spatial_width
                )
            progress.update(1)
            
            # 5. Apply room acoustics if specified
            processed_audio = stereo_audio
            if config.room_simulation and self.room_simulation:
                logger.info(f"Applying {config.room_size} room acoustics...")
                progress.update(0, status_info={"current_operation": "Applying room acoustics"})
                
                processed_audio = self.room_processor.apply_room_acoustics(
                    stereo_audio, room_size=config.room_size
                )
            progress.update(1)
            
            # 6. Apply effects
            # Apply breathing modulation if specified
            if config.breathing_modulation:
                logger.info("Applying breathing rhythm modulation...")
                progress.update(0, status_info={"current_operation": "Applying breathing modulation"})
                
                breathing_params = BreathingModulation(**config.breathing_modulation)
                processed_audio = self.breathing_modulator.apply_breathing_modulation(
                    processed_audio, breathing_params
                )
            
            # Apply sleep cycle modulation if specified
            if config.sleep_cycle_modulation:
                logger.info("Applying sleep cycle modulation...")
                progress.update(0, status_info={"current_operation": "Applying sleep cycle modulation"})
                
                sleep_cycle_params = SleepCycleModulation(**config.sleep_cycle_modulation)
                processed_audio = self.sleep_cycle_modulator.apply_sleep_cycle_modulation(
                    processed_audio, sleep_cycle_params
                )
            
            # Apply Moro reflex prevention if specified
            if config.moro_reflex_prevention:
                logger.info("Applying Moro reflex prevention...")
                progress.update(0, status_info={"current_operation": "Applying Moro reflex prevention"})
                
                reflex_params = MoroReflexPrevention(**config.moro_reflex_prevention)
                processed_audio = self.reflex_preventer.apply_moro_reflex_prevention(
                    processed_audio, reflex_params
                )
            
            # Apply dynamic volume if specified
            if config.dynamic_volume:
                logger.info("Applying dynamic volume adjustment...")
                progress.update(0, status_info={"current_operation": "Applying dynamic volume"})
                
                dynamic_volume_params = DynamicVolume(**config.dynamic_volume)
                processed_audio = self.dynamic_volume_processor.apply_dynamic_volume(
                    processed_audio, dynamic_volume_params
                )
            
            progress.update(1)
            
            # 7. Create seamless loop
            logger.info("Creating seamless loop...")
            progress.update(0, status_info={"current_operation": "Creating seamless loop"})
            looped = self.audio_processor.create_seamless_loop(processed_audio, self.crossfade_duration)
            progress.update(1)
            
            # 8. Add fades
            logger.info("Adding fade in/out...")
            progress.update(0, status_info={"current_operation": "Adding fades"})
            faded = self.audio_processor.add_fade(looped)
            progress.update(1)
            
            # 9. Apply loudness normalization
            logger.info("Applying EBU R128 loudness normalization...")
            progress.update(0, status_info={"current_operation": "Applying loudness normalization"})
            normalized = self.loudness_normalizer.apply_ebu_r128_normalization(faded)
            progress.update(1)
            
            # 10. Export to file
            # Generate output filename if not provided
            if not hasattr(config, "output_file") or not config.output_file:
                # Create filename from configuration
                base_name = config.name.lower().replace(" ", "_")
                output_file = f"{base_name}_{int(config.duration_seconds / 3600)}h.{config.output_format}"
            else:
                output_file = config.output_file
                
                # Ensure correct extension
                if not output_file.lower().endswith(f".{config.output_format.lower()}"):
                    output_file = f"{os.path.splitext(output_file)[0]}.{config.output_format.lower()}"
            
            # Create directory if needed
            try:
                os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
            except OSError as e:
                logger.error(f"Error creating output directory: {e}")
                # Use current directory as fallback
                output_file = os.path.basename(output_file)
            
            logger.info(f"Saving to {config.output_format.upper()} file...")
            progress.update(0, status_info={"current_operation": f"Exporting to {config.output_format}"})
            
            # Save to file in appropriate format
            if config.output_format.lower() == OutputFormat.MP3.value:
                output_path = self.audio_exporter.save_to_mp3(normalized, output_file)
            else:
                output_path = self.audio_exporter.save_to_wav(
                    normalized, output_file, config.volume
                )
            
            if output_path is None:
                logger.error("Failed to save output file")
                raise RuntimeError("Failed to save output file")
            
            # Generate visualization if requested
            if config.render_visualization:
                logger.info("Generating frequency spectrum visualization...")
                progress.update(0, status_info={"current_operation": "Generating visualization"})
                
                viz_path = os.path.splitext(output_file)[0] + "_spectrum.png"
                self.spectrum_visualizer.visualize_spectrum(
                    normalized, config.name, viz_path
                )
            
            # Print safety information based on volume
            self.output_manager.print_safety_information(config.volume)
            
            # Mark as complete
            progress.complete(status_info={
                "output_path": output_path,
                "duration_seconds": duration_seconds,
                "sample_rate": config.sample_rate,
                "bit_depth": config.bit_depth
            })
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating sound from config: {e}")
            # Clean up partial files
            if output_path and os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except OSError:
                    pass
                    
            # Clean up any temporary files
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except OSError:
                        pass
                        
            # Report failure
            if progress:
                progress.complete(status_info={"error": str(e)})
                
            return None

    def generate_from_profile(
        self,
        problem: str,
        duration_hours: float,
        volume: Optional[float] = None,
        output_file: Optional[str] = None,
        visualize: bool = False,
        format: str = "wav",
        progress_callback = None,
    ) -> Optional[str]:
        """
        Generate a sound file based on a predefined problem profile with enhanced quality
        
        Args:
            problem: Name of the problem profile to use
            duration_hours: Length of generated audio in hours
            volume: Output volume level (0.0-1.0), uses profile default if None
            output_file: Path to save the output file
            visualize: Whether to generate spectrum visualization
            format: Output format ('wav' or 'mp3')
            progress_callback: Optional callback for progress updates
            
        Returns:
            Path to the generated audio file or None if generation failed
        """
        if problem not in self.profile_manager.problem_profiles:
            error_msg = f"Unknown problem profile: {problem}. Available profiles: {list(self.profile_manager.problem_profiles.keys())}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        try:
            profile = self.profile_manager.problem_profiles[problem]

            # Use provided volume or default from profile
            if volume is None:
                volume = profile.recommended_volume

            # Create progress reporter
            total_steps = 10  # Approximate number of major steps
            progress = ProgressReporter(
                total_steps, 
                f"Generating {problem} profile", 
                progress_callback
            )

            # Calculate duration in seconds
            duration_seconds = int(duration_hours * 3600)

            # Generate primary noise
            logger.info(f"Generating {profile.primary_noise} noise...")
            progress.update(0, force=True, status_info={
                "current_operation": f"Generating {profile.primary_noise} noise"
            })
            primary_noise = self.sound_profiles[profile.primary_noise](duration_seconds)
            progress.update(1)

            # Apply frequency shaping with additional parameters
            logger.info(f"Applying {profile.frequency_focus} frequency shaping...")
            progress.update(0, status_info={"current_operation": "Applying frequency shaping"})
            
            # Extract additional frequency parameters if present
            additional_params = {}
            if profile.frequency_emphasis is not None:
                additional_params["frequency_emphasis"] = profile.frequency_emphasis
            if profile.low_pass_filter is not None:
                additional_params["low_pass_filter"] = profile.low_pass_filter
            if profile.frequency_limiting is not None:
                additional_params["frequency_limiting"] = profile.frequency_limiting
            if profile.circadian_alignment is not None:
                additional_params["circadian_alignment"] = profile.circadian_alignment

            shaped_noise = self.frequency_processor.apply_frequency_shaping(
                primary_noise, profile.frequency_focus, additional_params
            )
            progress.update(1)

            # Generate overlay sounds if specified
            overlay_sounds = []
            mix_ratios = []

            if profile.overlay_sounds:
                logger.info("Generating overlay sounds...")
                progress.update(0, status_info={
                    "current_operation": f"Generating overlay sounds"
                })
                
                # Create tasks for parallel processing
                overlay_tasks = []
                
                for overlay_type in profile.overlay_sounds:
                    if overlay_type == "heartbeat":
                        if profile.heartbeat_parameters is not None:
                            overlay_tasks.append((
                                self.natural_generator.generate_heartbeat,
                                {
                                    "duration_seconds": duration_seconds,
                                    "heartbeat_params": profile.heartbeat_parameters
                                }
                            ))
                        else:
                            overlay_tasks.append((
                                self.natural_generator.generate_heartbeat,
                                {"duration_seconds": duration_seconds}
                            ))
                        mix_ratios.append(0.2)  # 20% mix for heartbeat
                    elif overlay_type == "shushing":
                        overlay_tasks.append((
                            self.natural_generator.generate_shushing_sound,
                            {"duration_seconds": duration_seconds}
                        ))
                        mix_ratios.append(0.3)  # 30% mix for shushing
                    elif overlay_type == "umbilical_swish":
                        overlay_tasks.append((
                            self.womb_generator.generate_umbilical_swish,
                            {"duration_seconds": duration_seconds}
                        ))
                        mix_ratios.append(0.25)  # 25% mix for umbilical swish
                
                # Process in parallel if multiple overlays
                if len(overlay_tasks) > 1:
                    overlay_results = self.parallel_processor.process(
                        overlay_tasks, 
                        description="Generate overlay sounds"
                    )
                    overlay_sounds = [result for result in overlay_results if result is not None]
                elif len(overlay_tasks) == 1:
                    # Just process directly for a single overlay
                    func, kwargs = overlay_tasks[0]
                    overlay_sounds = [func(**kwargs)]
                
                # Mix sounds if we have overlays
                if overlay_sounds:
                    logger.info("Mixing sounds...")
                    progress.update(0, status_info={"current_operation": "Mixing sounds"})
                    
                    mixed = self.audio_processor.mix_sounds(
                        shaped_noise, overlay_sounds, mix_ratios, profile.motion_smoothing
                    )
                else:
                    mixed = shaped_noise
            else:
                mixed = shaped_noise
            
            progress.update(1)

            # Apply stereo processing
            logger.info("Applying spatial processing...")
            progress.update(0, status_info={"current_operation": "Applying spatial processing"})
            spatial_width = profile.spatial_width

            # Apply HRTF-based spatialization or basic stereo widening
            if len(mixed.shape) == 1:  # Only apply if mono
                stereo_audio = self.spatial_processor.apply_hrtf_spatialization(mixed, width=spatial_width)
            else:
                # Already stereo, just adjust width
                stereo_audio = self.spatial_processor.apply_basic_stereo_widening(mixed, width=spatial_width)
            progress.update(1)

            # Apply room acoustics if specified
            processed_audio = stereo_audio
            if self.room_simulation:
                logger.info(f"Applying {profile.room_size} room acoustics...")
                progress.update(0, status_info={"current_operation": "Applying room acoustics"})
                
                processed_audio = self.room_processor.apply_room_acoustics(
                    stereo_audio, room_size=profile.room_size
                )
            progress.update(1)

            # Apply enhanced features based on profile type
            logger.info("Applying profile-specific effects...")
            progress.update(0, status_info={"current_operation": "Applying profile effects"})

            # Apply Moro reflex prevention for startle_reflex profile
            if profile.moro_reflex_prevention is not None:
                logger.info("Applying Moro reflex prevention...")
                processed_audio = self.reflex_preventer.apply_moro_reflex_prevention(
                    processed_audio, profile.moro_reflex_prevention
                )

            # Apply sleep cycle modulation
            if profile.sleep_cycle_modulation is not None:
                logger.info("Applying sleep cycle modulation...")
                processed_audio = self.sleep_cycle_modulator.apply_sleep_cycle_modulation(
                    processed_audio, profile.sleep_cycle_modulation
                )

            # Apply dynamic shushing for colic profiles if shushing is in overlay_sounds
            if profile.dynamic_shushing is not None and "shushing" in profile.overlay_sounds:
                logger.info("Applying dynamic shushing...")
                # Get the shushing component index
                shushing_idx = profile.overlay_sounds.index("shushing")
                if shushing_idx < len(overlay_sounds):
                    # Apply dynamic shushing if we have the shushing overlay
                    shushing_sound = overlay_sounds[shushing_idx]
                    processed_audio = self.natural_generator.apply_dynamic_shushing(
                        processed_audio, shushing_sound, profile.dynamic_shushing
                    )

            # Apply breathing modulation
            if profile.breathing_modulation is not None:
                logger.info("Applying breathing rhythm modulation...")
                processed_audio = self.breathing_modulator.apply_breathing_modulation(
                    processed_audio, profile.breathing_modulation
                )

            # Apply parental voice if enabled
            if profile.parental_voice is not None:
                logger.info("Applying parental voice overlay...")
                processed_audio = self.natural_generator.apply_parental_voice(
                    processed_audio, profile.parental_voice
                )

            # Apply auto shutoff for safety
            if profile.safety_features is not None:
                logger.info("Applying safety auto-shutoff...")
                processed_audio = self.dynamic_volume_processor.apply_auto_shutoff(
                    processed_audio, profile.safety_features
                )

            # Apply dynamic volume adjustment (for newborn transition)
            if profile.dynamic_volume is not None:
                logger.info("Applying dynamic volume adjustment...")
                processed_audio = self.dynamic_volume_processor.apply_dynamic_volume(
                    processed_audio, profile.dynamic_volume
                )
            
            progress.update(1)

            # Create seamless looping
            logger.info("Creating seamless loop...")
            progress.update(0, status_info={"current_operation": "Creating seamless loop"})
            looped = self.audio_processor.create_seamless_loop(processed_audio, self.crossfade_duration)
            progress.update(1)

            # Add fade in/out
            logger.info("Adding fade in/out...")
            progress.update(0, status_info={"current_operation": "Adding fades"})
            faded = self.audio_processor.add_fade(looped)
            progress.update(1)

            # Generate output filename if not provided
            if output_file is None:
                output_file = f"{problem}_noise_{int(duration_hours)}hours.{format}"

            # Apply EBU R128 loudness normalization
            logger.info("Applying EBU R128 loudness normalization...")
            progress.update(0, status_info={"current_operation": "Applying loudness normalization"})
            normalized = self.loudness_normalizer.apply_ebu_r128_normalization(faded)
            progress.update(1)

            # Save to file in appropriate format
            logger.info(f"Saving to {format.upper()} file...")
            progress.update(0, status_info={"current_operation": f"Exporting to {format}"})
            
            # Create directory if needed
            try:
                os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
            except OSError as e:
                logger.error(f"Error creating output directory: {e}")
                # Use current directory as fallback
                output_file = os.path.basename(output_file)
            
            if format.lower() == "mp3":
                output_path = self.audio_exporter.save_to_mp3(normalized, output_file)
            else:
                output_path = self.audio_exporter.save_to_wav(normalized, output_file, volume)

            if not output_path:
                logger.error("Failed to save output file")
                raise RuntimeError("Failed to save output file")

            # Visualize if requested
            if visualize:
                logger.info("Generating frequency spectrum visualization...")
                progress.update(0, status_info={"current_operation": "Generating visualization"})
                
                viz_path = os.path.splitext(output_file)[0] + "_spectrum.png"
                self.spectrum_visualizer.visualize_spectrum(normalized, f"{problem} Noise Profile", viz_path)

            logger.info(f"Generated {format.upper()} file: {output_path}")

            # Print safety information based on volume
            self.output_manager.print_safety_information(volume)

            # Mark as complete
            progress.complete(status_info={
                "output_path": output_path,
                "duration_hours": duration_hours,
                "problem": problem
            })
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating profile {problem}: {e}")
            # Clean up any partial files
            if output_file and os.path.exists(output_file):
                try:
                    os.remove(output_file)
                except OSError:
                    pass
            # Report failure
            if 'progress' in locals():
                progress.complete(status_info={"error": str(e)})
            return None

    def generate_custom(
        self,
        primary_noise: str,
        overlay_sounds: List[str],
        frequency_focus: FrequencyFocus,
        duration_hours: float,
        volume: float = 0.6,
        spatial_width: float = 0.5,
        room_size: RoomSize = RoomSize.MEDIUM,
        output_file: Optional[str] = None,
        visualize: bool = False,
        format: str = "wav",
        # Advanced parameters
        variable_heartbeat: bool = False,
        heartbeat_bpm_range: Tuple[float, float] = (60.0, 80.0),
        motion_smoothing: bool = False,
        breathing_modulation: bool = False,
        breathing_rate: float = Constants.BREATHING_RATE_CPM,
        dynamic_volume: bool = False,
        frequency_emphasis_params: Optional[Dict[str, Any]] = None,
        low_pass_filter_hz: Optional[float] = None,
        # New noise enhancement parameters
        use_equal_loudness: bool = True,
        use_limiter: bool = True,
        use_organic_drift: bool = True,
        use_diffusion: bool = False,
        progress_callback = None,
    ) -> Optional[str]:
        """
        Generate a custom sound profile with enhanced quality and optional advanced parameters
        
        Args:
            primary_noise: Type of primary noise to generate
            overlay_sounds: List of overlay sounds to mix in
            frequency_focus: Focus of frequency shaping
            duration_hours: Length of generated audio in hours
            volume: Output volume level (0.0-1.0)
            spatial_width: Stereo width (0.0-1.0)
            room_size: Size of room acoustics to simulate
            output_file: Path to save the output file
            visualize: Whether to generate spectrum visualization
            format: Output format ('wav' or 'mp3')
            variable_heartbeat: Whether to use variable heartbeat rate
            heartbeat_bpm_range: Range of BPM variation (min, max)
            motion_smoothing: Whether to apply motion smoothing
            breathing_modulation: Whether to apply breathing modulation
            breathing_rate: Breathing rate in cycles per minute
            dynamic_volume: Whether to apply dynamic volume
            frequency_emphasis_params: Custom parameters for frequency emphasis
            low_pass_filter_hz: Cutoff frequency for low-pass filter
            progress_callback: Optional callback for progress updates
            
        Returns:
            Path to the generated audio file or None if generation failed
        """
        # Check if primary noise type is supported
        if primary_noise not in self.sound_profiles:
            error_msg = f"Unknown noise type: {primary_noise}. Available types: {list(self.sound_profiles.keys())}"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        # Pass new noise enhancement parameters to the noise generator if applicable
        if primary_noise in ["white", "pink", "brown"]:
            self.noise_generator.use_equal_loudness = use_equal_loudness
            self.noise_generator.use_limiter = use_limiter
            self.noise_generator.use_organic_drift = use_organic_drift
            self.noise_generator.use_diffusion = use_diffusion

        # Create progress reporter
        total_steps = 10  # Approximate number of major steps
        progress = ProgressReporter(
            total_steps,
            f"Generating custom {primary_noise} sound",
            progress_callback
        )

        try:
            # Duration in seconds
            duration_seconds = int(duration_hours * 3600)

            # Generate primary noise
            logger.info(f"Generating {primary_noise} noise...")
            progress.update(0, force=True, status_info={
                "current_operation": f"Generating {primary_noise} noise"
            })
            primary = self.sound_profiles[primary_noise](duration_seconds)
            progress.update(1)

            # Create additional parameters for frequency shaping
            additional_params = {}

            # Add frequency emphasis if requested
            if frequency_emphasis_params is not None:
                emphasis = FrequencyEmphasis(
                    enabled=True,
                    center_hz=frequency_emphasis_params.get("center_hz", 500.0),
                    bandwidth_hz=frequency_emphasis_params.get("bandwidth_hz", 200.0),
                    gain_db=frequency_emphasis_params.get("gain_db", 3.0),
                )
                additional_params["frequency_emphasis"] = emphasis

            # Add low pass filter if requested
            if low_pass_filter_hz is not None:
                lpf = LowPassFilter(enabled=True, cutoff_hz=low_pass_filter_hz)
                additional_params["low_pass_filter"] = lpf

            # Apply frequency shaping
            logger.info(f"Applying {frequency_focus} frequency shaping...")
            progress.update(0, status_info={"current_operation": "Applying frequency shaping"})
            shaped = self.frequency_processor.apply_frequency_shaping(primary, frequency_focus, additional_params)
            progress.update(1)

            # Generate overlay sounds
            overlay_arrays = []
            mix_ratios = []

            if overlay_sounds:
                logger.info("Generating overlay sounds...")
                progress.update(0, status_info={
                    "current_operation": f"Generating overlay sounds"
                })
                
                # Create tasks for parallel processing
                overlay_tasks = []
                
                for overlay in overlay_sounds:
                    if overlay in self.sound_profiles:
                        logger.info(f"Adding {overlay} overlay...")

                        # Special case for heartbeat with variable rate
                        if overlay == "heartbeat" and variable_heartbeat:
                            heartbeat_params = HeartbeatParameters(
                                variable_rate=True,
                                bpm_range=heartbeat_bpm_range,
                            )
                            overlay_tasks.append((
                                self.natural_generator.generate_heartbeat,
                                {
                                    "duration_seconds": duration_seconds,
                                    "heartbeat_params": heartbeat_params
                                }
                            ))
                        else:
                            overlay_tasks.append((
                                self.sound_profiles[overlay],
                                {"duration_seconds": duration_seconds}
                            ))

                        # Default mix ratios
                        if overlay == "heartbeat":
                            mix_ratios.append(0.2)  # 20% heartbeat
                        elif overlay == "shushing":
                            mix_ratios.append(0.3)  # 30% shushing
                        elif overlay == "umbilical_swish":
                            mix_ratios.append(0.25)  # 25% umbilical swish
                        else:
                            mix_ratios.append(0.2)  # 20% default
                
                # Process in parallel if multiple overlays
                if len(overlay_tasks) > 1:
                    overlay_results = self.parallel_processor.process(
                        overlay_tasks, 
                        description="Generate overlay sounds"
                    )
                    overlay_arrays = [result for result in overlay_results if result is not None]
                elif len(overlay_tasks) == 1:
                    # Just process directly for a single overlay
                    func, kwargs = overlay_tasks[0]
                    overlay_arrays = [func(**kwargs)]
                
            progress.update(1)

            # Create motion smoothing parameters if requested
            motion_params = None
            if motion_smoothing:
                motion_params = MotionSmoothing(enabled=True, transition_seconds=5.0)

            # Mix sounds
            if overlay_arrays:
                logger.info("Mixing sounds...")
                progress.update(0, status_info={"current_operation": "Mixing sounds"})
                mixed = self.audio_processor.mix_sounds(shaped, overlay_arrays, mix_ratios, motion_params)
            else:
                mixed = shaped
            progress.update(1)

            # Apply stereo processing
            logger.info("Applying spatial processing...")
            progress.update(0, status_info={"current_operation": "Applying spatial processing"})
            
            # Apply HRTF-based spatialization or basic stereo widening
            if len(mixed.shape) == 1:  # Only apply if mono
                stereo_audio = self.spatial_processor.apply_hrtf_spatialization(mixed, width=spatial_width)
            else:
                # Already stereo, just adjust width
                stereo_audio = self.spatial_processor.apply_basic_stereo_widening(mixed, width=spatial_width)
            progress.update(1)

            # Apply room acoustics if specified
            processed_audio = stereo_audio
            if self.room_simulation:
                logger.info(f"Applying {room_size} room acoustics...")
                progress.update(0, status_info={"current_operation": "Applying room acoustics"})
                processed_audio = self.room_processor.apply_room_acoustics(stereo_audio, room_size=room_size)
            progress.update(1)

            # Apply effects
            logger.info("Applying effects...")
            progress.update(0, status_info={"current_operation": "Applying effects"})
            
            # Apply breathing modulation if requested
            if breathing_modulation:
                logger.info("Applying breathing rhythm modulation...")
                breathing_params = BreathingModulation(enabled=True, cycles_per_minute=breathing_rate)
                processed_audio = self.breathing_modulator.apply_breathing_modulation(processed_audio, breathing_params)

            # Apply dynamic volume reduction if requested
            if dynamic_volume:
                logger.info("Applying dynamic volume adjustment...")
                dynamic_volume_params = DynamicVolume(
                    enabled=True,
                    initial_db=65.0,
                    reduction_db=50.0,
                    reduction_time_minutes=15.0,
                    fade_duration_seconds=180.0,  # 3 minutes
                )
                processed_audio = self.dynamic_volume_processor.apply_dynamic_volume(processed_audio, dynamic_volume_params)
            progress.update(1)

            # Create seamless loop
            logger.info("Creating seamless loop...")
            progress.update(0, status_info={"current_operation": "Creating seamless loop"})
            looped = self.audio_processor.create_seamless_loop(processed_audio)
            progress.update(1)

            # Add fades
            logger.info("Adding fade in/out...")
            progress.update(0, status_info={"current_operation": "Adding fades"})
            faded = self.audio_processor.add_fade(looped)
            progress.update(1)

            # Generate output filename if not provided
            if output_file is None:
                overlay_str = "_".join(overlay_sounds) if overlay_sounds else "no_overlay"
                output_file = f"custom_{primary_noise}_{overlay_str}_{int(duration_hours)}hours.{format}"

            # Apply EBU R128 loudness normalization
            logger.info("Applying EBU R128 loudness normalization...")
            progress.update(0, status_info={"current_operation": "Applying loudness normalization"})
            normalized = self.loudness_normalizer.apply_ebu_r128_normalization(faded)
            progress.update(1)

            # Save to file in appropriate format
            logger.info(f"Saving to {format.upper()} file...")
            progress.update(0, status_info={"current_operation": f"Exporting to {format}"})
            
            # Create directory if needed
            try:
                os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
            except OSError as e:
                logger.error(f"Error creating output directory: {e}")
                # Use current directory as fallback
                output_file = os.path.basename(output_file)
            
            if format.lower() == "mp3":
                output_path = self.audio_exporter.save_to_mp3(normalized, output_file)
            else:
                output_path = self.audio_exporter.save_to_wav(normalized, output_file, volume)

            if not output_path:
                logger.error("Failed to save output file")
                raise RuntimeError("Failed to save output file")

            # Visualize if requested
            if visualize:
                logger.info("Generating frequency spectrum visualization...")
                progress.update(0, status_info={"current_operation": "Generating visualization"})
                viz_path = os.path.splitext(output_file)[0] + "_spectrum.png"
                self.spectrum_visualizer.visualize_spectrum(normalized, f"Custom {primary_noise} Noise Profile", viz_path)

            # Print safety information
            self.output_manager.print_safety_information(volume)

            logger.info(f"Done! Generated {format.upper()} file: {output_path}")
            
            # Mark as complete
            progress.complete(status_info={
                "output_path": output_path,
                "duration_hours": duration_hours,
                "primary_noise": primary_noise
            })
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating custom sound: {e}")
            # Clean up any partial files
            if 'output_file' in locals() and output_file and os.path.exists(output_file):
                try:
                    os.remove(output_file)
                except OSError:
                    pass
            # Report failure
            if 'progress' in locals():
                progress.complete(status_info={"error": str(e)})
            return None

    @classmethod
    def get_problem_descriptions(cls) -> Dict[str, str]:
        """Return descriptions of all predefined problem profiles"""
        try:
            generator = cls()
            return generator.profile_manager.get_profile_descriptions()
        except Exception as e:
            logger.error(f"Error getting problem descriptions: {e}")
            return {}

    @classmethod
    def get_available_sounds(cls) -> List[str]:
        """Return list of available sound types"""
        try:
            generator = cls()
            return list(generator.sound_profiles.keys())
        except Exception as e:
            logger.error(f"Error getting available sounds: {e}")
            return []