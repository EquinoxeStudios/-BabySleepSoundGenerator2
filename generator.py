"""
Core generator class that orchestrates sound generation and processing.
"""

import os
import random
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from models.constants import Constants, FrequencyFocus, RoomSize, LoopingMethod, OutputFormat
from models.parameters import (
    HeartbeatParameters, FrequencyEmphasis, LowPassFilter, FrequencyLimiting, CircadianAlignment,
    DynamicVolume, MotionSmoothing, MoroReflexPrevention, SleepCycleModulation, 
    DynamicShushing, BreathingModulation, SafetyFeatures, ParentalVoice, ProblemProfile
)

from sound_profiles.noise import NoiseGenerator
from sound_profiles.natural import NaturalSoundGenerator
from sound_profiles.womb import WombSoundGenerator

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

from utils.optional_imports import HAS_PERLIN, HAS_LOUDNORM, HAS_LIBROSA, HAS_PYROOMACOUSTICS
from utils.perlin_utils import generate_perlin_noise, apply_modulation, generate_dynamic_modulation

logger = logging.getLogger("BabySleepSoundGenerator")


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
    ):
        """Initialize the generator with professional audio quality settings"""
        self.sample_rate = sample_rate
        self.bit_depth = bit_depth  # 16, 24, or 32
        self.channels = channels
        self.target_loudness = target_loudness  # LUFS
        self.use_hrtf = use_hrtf and HAS_LIBROSA
        self.room_simulation = room_simulation and HAS_PYROOMACOUSTICS

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

        # Initialize component generators
        self.noise_generator = NoiseGenerator(sample_rate, self.use_perlin, modulation_depth=self.modulation_depth)
        self.natural_generator = NaturalSoundGenerator(sample_rate, self.use_perlin)
        self.womb_generator = WombSoundGenerator(sample_rate, self.use_perlin)

        # Initialize processors
        self.spatial_processor = SpatialProcessor(sample_rate, use_hrtf)
        self.room_processor = RoomAcousticsProcessor(sample_rate, room_simulation)
        self.frequency_processor = FrequencyProcessor(sample_rate)
        self.modulation_processor = ModulationProcessor(
            sample_rate, self.use_perlin, depth=self.modulation_depth, rate=self.modulation_rate
        )
        
        # Initialize effect modules
        self.breathing_modulator = BreathingModulator(sample_rate, use_perlin)
        self.sleep_cycle_modulator = SleepCycleModulator(sample_rate, use_perlin)
        self.dynamic_volume_processor = DynamicVolumeProcessor(sample_rate, self.volume_to_db_spl)
        self.reflex_preventer = ReflexPreventer(sample_rate)
        
        # Initialize output modules
        self.loudness_normalizer = LoudnessNormalizer(sample_rate, target_loudness)
        self.audio_exporter = AudioExporter(sample_rate, bit_depth, channels)
        self.spectrum_visualizer = SpectrumVisualizer(sample_rate)
        
        # Define available sound profiles
        self.sound_profiles = {
            "white": lambda duration, **kwargs: self.noise_generator.generate(duration, sound_type="white", **kwargs),
            "pink": lambda duration, **kwargs: self.noise_generator.generate(duration, sound_type="pink", **kwargs),
            "brown": lambda duration, **kwargs: self.noise_generator.generate(duration, sound_type="brown", **kwargs),
            "womb": lambda duration, **kwargs: self.womb_generator.generate(duration, sound_type="womb", **kwargs),
            "heartbeat": lambda duration, **kwargs: self.natural_generator.generate(duration, sound_type="heartbeat", **kwargs),
            "shushing": lambda duration, **kwargs: self.natural_generator.generate(duration, sound_type="shushing", **kwargs),
            "fan": lambda duration, **kwargs: self.natural_generator.generate(duration, sound_type="fan", **kwargs),
            "umbilical_swish": lambda duration, **kwargs: self.womb_generator.generate(duration, sound_type="umbilical_swish", **kwargs),
        }

        # Predefined baby problems and recommended profiles with enhanced scientific parameters
        self._initialize_problem_profiles()

    def _initialize_problem_profiles(self) -> None:
        """Initialize problem profiles using dataclasses for better structure"""
        from models.profiles import create_problem_profiles
        self.problem_profiles = create_problem_profiles()

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

    def mix_sounds(
        self,
        primary: np.ndarray,
        overlays: List[np.ndarray],
        mix_ratios: List[float],
        motion_smoothing: Optional[MotionSmoothing] = None,
    ) -> np.ndarray:
        """
        Mix multiple sound arrays together with specified mix ratios.

        Args:
            primary: Primary sound array
            overlays: List of overlay sound arrays
            mix_ratios: List of mix ratios for each overlay
            motion_smoothing: Optional parameters for smooth transitions

        Returns:
            Mixed audio array
        """
        if not overlays or not mix_ratios:
            return primary

        # Validate input
        if len(overlays) != len(mix_ratios):
            raise ValueError("Number of overlays must match number of mix ratios")
            
        # Check if primary is stereo
        is_stereo = len(primary.shape) > 1

        # Start with the primary sound
        mixed = primary.copy()

        # Add each overlay with its mix ratio
        for overlay, ratio in zip(overlays, mix_ratios):
            # Convert overlay to stereo if needed
            if len(overlay.shape) == 1 and is_stereo:
                overlay_stereo = np.zeros((len(overlay), primary.shape[1]))
                for c in range(primary.shape[1]):
                    overlay_stereo[:, c] = overlay
                overlay = overlay_stereo
            elif len(overlay.shape) > 1 and not is_stereo:
                # Take only the first channel if primary is mono
                overlay = overlay[:, 0]

            # Make sure the overlay is the same length as the primary
            if len(overlay) > len(mixed):
                overlay = overlay[: len(mixed)]
            elif len(overlay) < len(mixed):
                # Pad with zeros if necessary
                if is_stereo:
                    overlay = np.pad(overlay, ((0, len(mixed) - len(overlay)), (0, 0)))
                else:
                    overlay = np.pad(overlay, (0, len(mixed) - len(overlay)))

            # Apply motion smoothing if requested
            if motion_smoothing is not None and motion_smoothing.enabled:
                # Create a smooth crossfade envelope for the overlay
                transition_samples = int(motion_smoothing.transition_seconds * self.sample_rate)

                # Ensure transition isn't too long compared to audio
                if transition_samples * 2 > len(mixed):
                    transition_samples = len(mixed) // 4

                # Create a smooth envelope that gradually increases the overlay
                if is_stereo:
                    envelope = np.ones((len(mixed), primary.shape[1])) * ratio
                    
                    # Apply gradual fade-in at the beginning
                    fade_in = np.linspace(0, 1, transition_samples)
                    envelope[:transition_samples] *= fade_in[:, np.newaxis]
                    
                    # Apply gradual fade-out at the end
                    fade_out = np.linspace(1, 0, transition_samples)
                    envelope[-transition_samples:] *= fade_out[:, np.newaxis]
                else:
                    envelope = np.ones(len(mixed)) * ratio
                    
                    # Apply gradual fade-in at the beginning
                    fade_in = np.linspace(0, 1, transition_samples)
                    envelope[:transition_samples] *= fade_in
                    
                    # Apply gradual fade-out at the end
                    fade_out = np.linspace(1, 0, transition_samples)
                    envelope[-transition_samples:] *= fade_out

                # Apply the smooth mix
                mixed = mixed * (1 - envelope) + overlay * envelope
            else:
                # Standard mixing with constant ratio
                mixed = mixed * (1 - ratio) + overlay * ratio

        # Normalize to prevent clipping
        max_val = np.max(np.abs(mixed))
        if max_val > Constants.MAX_AUDIO_VALUE:
            mixed = mixed / max_val * Constants.MAX_AUDIO_VALUE

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
        # Don't try to analyze too much data - use segments
        segment_samples = int(segment_duration * self.sample_rate)

        # We'll look for matches between beginning and end segments
        if len(audio) <= 2 * segment_samples:
            # If audio is too short, just use middle point
            return len(audio) // 2

        # For stereo, analyze first channel only
        if len(audio.shape) > 1:
            analysis_channel = audio[:, 0]
        else:
            analysis_channel = audio

        # Get beginning segment
        begin_segment = analysis_channel[:segment_samples]

        # Try different positions near the end to find best correlation
        best_correlation = -np.inf
        best_position = len(analysis_channel) - segment_samples

        # Search the last 20% of the audio for good matching points
        search_start = int(len(analysis_channel) * 0.8)

        # To make search faster, we'll check candidates at regular intervals
        step = self.sample_rate // 10  # Check every 100ms

        # First pass: coarse search
        for pos in range(search_start, len(analysis_channel) - segment_samples, step):
            end_segment = analysis_channel[pos : pos + segment_samples]

            # Calculate cross-correlation to find similarity
            correlation = np.max(np.correlate(begin_segment, end_segment, 'valid'))

            if correlation > best_correlation:
                best_correlation = correlation
                best_position = pos

        # Second pass: fine-tune by checking neighboring points
        fine_range = self.sample_rate // 2  # Check +/- 500ms around best point
        for pos in range(
            max(search_start, best_position - fine_range),
            min(len(analysis_channel) - segment_samples, best_position + fine_range),
        ):
            end_segment = analysis_channel[pos : pos + segment_samples]
            correlation = np.max(np.correlate(begin_segment, end_segment, 'valid'))

            if correlation > best_correlation:
                best_correlation = correlation
                best_position = pos

        return best_position

    def create_seamless_loop(
        self, audio: np.ndarray, crossfade_seconds: float = None
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
            
        if self.looping_method == LoopingMethod.CROSS_CORRELATION:
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

        # Apply crossfade efficiently using apply_modulation
        if is_stereo:
            # Create crossfade for stereo
            crossfaded = apply_modulation(end, fade_out) + apply_modulation(beginning, fade_in)
            
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

    def _print_safety_information(self, volume: float) -> None:
        """Print safety information based on the volume level"""
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

    def generate_from_profile(
        self,
        problem: str,
        duration_hours: float,
        volume: float = None,
        output_file: str = None,
        visualize: bool = False,
        format: str = "wav",
    ) -> str:
        """
        Generate a sound file based on a predefined problem profile with enhanced quality
        
        Args:
            problem: Name of the problem profile to use
            duration_hours: Length of generated audio in hours
            volume: Output volume level (0.0-1.0), uses profile default if None
            output_file: Path to save the output file
            visualize: Whether to generate spectrum visualization
            format: Output format ('wav' or 'mp3')
            
        Returns:
            Path to the generated audio file
        """
        if problem not in self.problem_profiles:
            raise ValueError(
                f"Unknown problem profile: {problem}. Available profiles: {list(self.problem_profiles.keys())}"
            )

        profile = self.problem_profiles[problem]

        # Use provided volume or default from profile
        if volume is None:
            volume = profile.recommended_volume

        # Calculate duration in seconds
        duration_seconds = int(duration_hours * 3600)

        # Generate primary noise
        logger.info(f"Generating {profile.primary_noise} noise...")
        primary_noise = self.sound_profiles[profile.primary_noise](duration_seconds)

        # Apply frequency shaping with additional parameters
        logger.info(f"Applying {profile.frequency_focus} frequency shaping...")
        
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

        # Generate overlay sounds if specified
        overlay_sounds = []
        mix_ratios = []

        for overlay_type in profile.overlay_sounds:
            logger.info(f"Adding {overlay_type} overlay...")
            if overlay_type == "heartbeat":
                # Check for enhanced heartbeat parameters
                if profile.heartbeat_parameters is not None:
                    overlay = self.natural_generator.generate_heartbeat(
                        duration_seconds, profile.heartbeat_parameters
                    )
                else:
                    overlay = self.natural_generator.generate_heartbeat(duration_seconds)
                overlay_sounds.append(overlay)
                mix_ratios.append(0.2)  # 20% mix for heartbeat
            elif overlay_type == "shushing":
                overlay = self.natural_generator.generate_shushing_sound(duration_seconds)
                overlay_sounds.append(overlay)
                mix_ratios.append(0.3)  # 30% mix for shushing
            elif overlay_type == "umbilical_swish":
                overlay = self.womb_generator.generate_umbilical_swish(duration_seconds)
                overlay_sounds.append(overlay)
                mix_ratios.append(0.25)  # 25% mix for umbilical swish

        # Mix sounds together with motion smoothing if specified
        if overlay_sounds:
            logger.info("Mixing sounds...")
            mixed = self.mix_sounds(
                shaped_noise, overlay_sounds, mix_ratios, profile.motion_smoothing
            )
        else:
            mixed = shaped_noise

        # Apply stereo processing
        logger.info("Applying spatial processing...")
        spatial_width = profile.spatial_width

        # Apply HRTF-based spatialization or basic stereo widening
        if len(mixed.shape) == 1:  # Only apply if mono
            stereo_audio = self.spatial_processor.apply_hrtf_spatialization(mixed, width=spatial_width)
        else:
            # Already stereo, just adjust width
            stereo_audio = self.spatial_processor.apply_basic_stereo_widening(mixed, width=spatial_width)

        # Apply room acoustics if specified
        if self.room_simulation:
            logger.info(f"Applying {profile.room_size} room acoustics...")
            processed_audio = self.room_processor.apply_room_acoustics(
                stereo_audio, room_size=profile.room_size
            )
        else:
            processed_audio = stereo_audio

        # Apply enhanced features based on profile type

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

        # Create seamless looping
        logger.info("Creating seamless loop...")
        looped = self.create_seamless_loop(processed_audio, self.crossfade_duration)

        # Add fade in/out
        logger.info("Adding fade in/out...")
        faded = self.add_fade(looped)

        # Generate output filename if not provided
        if output_file is None:
            output_file = f"{problem}_noise_{int(duration_hours)}hours.{format}"

        # Apply EBU R128 loudness normalization
        logger.info("Applying EBU R128 loudness normalization...")
        normalized = self.loudness_normalizer.apply_ebu_r128_normalization(faded)

        # Save to file in appropriate format
        logger.info(f"Saving to {format.upper()} file...")
        if format.lower() == "mp3":
            output_path = self.audio_exporter.save_to_mp3(normalized, output_file)
        else:
            output_path = self.audio_exporter.save_to_wav(normalized, output_file, volume)

        # Visualize if requested
        if visualize:
            logger.info("Generating frequency spectrum visualization...")
            viz_path = os.path.splitext(output_file)[0] + "_spectrum.png"
            self.spectrum_visualizer.visualize_spectrum(normalized, f"{problem} Noise Profile", viz_path)

        logger.info(f"Generated {format.upper()} file: {output_path}")

        # Print safety information based on volume
        self._print_safety_information(volume)

        return output_path

    def generate_custom(
        self,
        primary_noise: str,
        overlay_sounds: List[str],
        frequency_focus: FrequencyFocus,
        duration_hours: float,
        volume: float = 0.6,
        spatial_width: float = 0.5,
        room_size: RoomSize = RoomSize.MEDIUM,
        output_file: str = None,
        visualize: bool = False,
        format: str = "wav",
        # Advanced parameters
        variable_heartbeat: bool = False,
        heartbeat_bpm_range: Tuple[float, float] = (60.0, 80.0),
        motion_smoothing: bool = False,
        breathing_modulation: bool = False,
        breathing_rate: float = Constants.BREATHING_RATE_CPM,
        dynamic_volume: bool = False,
        frequency_emphasis_params: Dict[str, Any] = None,
        low_pass_filter_hz: Optional[float] = None,
    ) -> str:
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
            
        Returns:
            Path to the generated audio file
        """
        # Check if primary noise type is supported
        if primary_noise not in self.sound_profiles:
            raise ValueError(
                f"Unknown noise type: {primary_noise}. Available types: {list(self.sound_profiles.keys())}"
            )

        # Duration in seconds
        duration_seconds = int(duration_hours * 3600)

        # Generate primary noise
        logger.info(f"Generating {primary_noise} noise...")
        primary = self.sound_profiles[primary_noise](duration_seconds)

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
        shaped = self.frequency_processor.apply_frequency_shaping(primary, frequency_focus, additional_params)

        # Generate overlay sounds
        overlay_arrays = []
        mix_ratios = []

        for overlay in overlay_sounds:
            if overlay in self.sound_profiles:
                logger.info(f"Adding {overlay} overlay...")

                # Special case for heartbeat with variable rate
                if overlay == "heartbeat" and variable_heartbeat:
                    heartbeat_params = HeartbeatParameters(
                        variable_rate=True,
                        bpm_range=heartbeat_bpm_range,
                    )
                    overlay_array = self.natural_generator.generate_heartbeat(duration_seconds, heartbeat_params)
                else:
                    overlay_array = self.sound_profiles[overlay](duration_seconds)

                overlay_arrays.append(overlay_array)

                # Default mix ratios
                if overlay == "heartbeat":
                    mix_ratios.append(0.2)  # 20% heartbeat
                elif overlay == "shushing":
                    mix_ratios.append(0.3)  # 30% shushing
                elif overlay == "umbilical_swish":
                    mix_ratios.append(0.25)  # 25% umbilical swish
                else:
                    mix_ratios.append(0.2)  # 20% default

        # Create motion smoothing parameters if requested
        motion_params = None
        if motion_smoothing:
            motion_params = MotionSmoothing(enabled=True, transition_seconds=5.0)

        # Mix sounds
        if overlay_arrays:
            logger.info("Mixing sounds...")
            mixed = self.mix_sounds(shaped, overlay_arrays, mix_ratios, motion_params)
        else:
            mixed = shaped

        # Apply stereo processing
        logger.info("Applying spatial processing...")
        # Apply HRTF-based spatialization or basic stereo widening
        if len(mixed.shape) == 1:  # Only apply if mono
            stereo_audio = self.spatial_processor.apply_hrtf_spatialization(mixed, width=spatial_width)
        else:
            # Already stereo, just adjust width
            stereo_audio = self.spatial_processor.apply_basic_stereo_widening(mixed, width=spatial_width)

        # Apply room acoustics if specified
        if self.room_simulation:
            logger.info(f"Applying {room_size} room acoustics...")
            processed_audio = self.room_processor.apply_room_acoustics(stereo_audio, room_size=room_size)
        else:
            processed_audio = stereo_audio

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

        # Create seamless loop
        logger.info("Creating seamless loop...")
        looped = self.create_seamless_loop(processed_audio)

        # Add fades
        logger.info("Adding fade in/out...")
        faded = self.add_fade(looped)

        # Generate output filename if not provided
        if output_file is None:
            overlay_str = "_".join(overlay_sounds) if overlay_sounds else "no_overlay"
            output_file = f"custom_{primary_noise}_{overlay_str}_{int(duration_hours)}hours.{format}"

        # Apply EBU R128 loudness normalization
        logger.info("Applying EBU R128 loudness normalization...")
        normalized = self.loudness_normalizer.apply_ebu_r128_normalization(faded)

        # Save to file in appropriate format
        logger.info(f"Saving to {format.upper()} file...")
        if format.lower() == "mp3":
            output_path = self.audio_exporter.save_to_mp3(normalized, output_file)
        else:
            output_path = self.audio_exporter.save_to_wav(normalized, output_file, volume)

        # Visualize if requested
        if visualize:
            logger.info("Generating frequency spectrum visualization...")
            viz_path = os.path.splitext(output_file)[0] + "_spectrum.png"
            self.spectrum_visualizer.visualize_spectrum(normalized, f"Custom {primary_noise} Noise Profile", viz_path)

        # Print safety information
        self._print_safety_information(volume)

        logger.info(f"Done! Generated {format.upper()} file: {output_path}")
        return output_path

    @classmethod
    def get_problem_descriptions(cls) -> Dict[str, str]:
        """Return descriptions of all predefined problem profiles"""
        generator = cls()
        return {
            problem: profile.description
            for problem, profile in generator.problem_profiles.items()
        }

    @classmethod
    def get_available_sounds(cls) -> List[str]:
        """Return list of available sound types"""
        generator = cls()
        return list(generator.sound_profiles.keys())