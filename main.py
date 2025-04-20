#!/usr/bin/env python
"""
BabySleepSoundGenerator - Professional Audio Quality White Noise Generator

This program generates customized noise profiles based on baby developmental stages and specific sleep problems.
It creates broadcast-standard audio files ready for YouTube or other platforms.

Enhanced with:
- Perlin/Simplex noise for more natural sound textures
- FFT-based spectral shaping for accurate noise colors
- Dynamic modulation to prevent listener fatigue
- Advanced seamless looping using cross-correlation
- EBU R128 loudness normalization
- ITU-R BS.1770 compliance
- Room impulse response simulations
- HRTF-based spatialization
- Developmental stage-specific sound profiles with research-backed parameters
- GPU-accelerated processing (if CuPy or PyTorch is available)
- Equal loudness filtering for better perceptual quality
- Organic micro-drift for more natural sound variations
- Soft-knee limiter for optimal levels without distortion
- Optional neural audio diffusion polish
"""

import argparse
import logging
import time
from pathlib import Path

from models.constants import Constants, FrequencyFocus, RoomSize, OutputFormat, NoiseColor
from generator import BabySleepSoundGenerator
from utils.logging import setup_logging

logger = logging.getLogger("BabySleepSoundGenerator")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Generate baby sleep sound profiles with broadcast audio quality"
    )

    # Main operation mode
    parser.add_argument(
        "--mode",
        choices=["predefined", "custom", "list"],
        default="predefined",
        help="Operation mode: use predefined problem profile, custom settings, or list available options",
    )

    # Predefined profile options
    parser.add_argument("--problem", type=str, help="Baby sleep problem to address")

    # Custom profile options
    parser.add_argument("--primary", type=str, help="Primary noise type")
    parser.add_argument(
        "--overlay", type=str, nargs="+", help="Overlay sound types to mix in"
    )
    parser.add_argument(
        "--focus",
        type=str,
        choices=[e.value for e in FrequencyFocus],
        help="Frequency focus",
    )
    parser.add_argument(
        "--spatial-width",
        type=float,
        default=0.5,
        help="Stereo width from 0.0 (mono) to 1.0 (maximum width)",
    )
    parser.add_argument(
        "--room-size",
        choices=[e.value for e in RoomSize],
        default=RoomSize.MEDIUM.value,
        help="Room size for acoustic simulation",
    )

    # Common options
    parser.add_argument("--duration", type=float, default=8.0, help="Duration in hours")
    parser.add_argument("--volume", type=float, help="Output volume (0.0-1.0)")
    parser.add_argument("--output", type=str, help="Output filename")
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate frequency spectrum visualization",
    )
    parser.add_argument(
        "--format", 
        choices=[e.value for e in OutputFormat], 
        default=OutputFormat.WAV.value, 
        help="Output file format"
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=Constants.DEFAULT_SAMPLE_RATE,
        help=f"Sample rate in Hz (default: {Constants.DEFAULT_SAMPLE_RATE})",
    )
    parser.add_argument(
        "--bit-depth",
        type=int,
        choices=[16, 24, 32],
        default=Constants.DEFAULT_BIT_DEPTH,
        help=f"Bit depth (default: {Constants.DEFAULT_BIT_DEPTH})",
    )
    parser.add_argument(
        "--channels",
        type=int,
        choices=[1, 2],
        default=Constants.DEFAULT_CHANNELS,
        help=f"Number of audio channels (default: {Constants.DEFAULT_CHANNELS})",
    )
    parser.add_argument(
        "--target-loudness",
        type=float,
        default=Constants.DEFAULT_TARGET_LOUDNESS,
        help=f"Target loudness in LUFS (default: {Constants.DEFAULT_TARGET_LOUDNESS} per EBU R128)",
    )

    # Advanced options
    parser.add_argument(
        "--disable-modulation", action="store_true", help="Disable dynamic modulation"
    )
    parser.add_argument(
        "--disable-hrtf", action="store_true", help="Disable HRTF-based spatialization"
    )
    parser.add_argument(
        "--disable-room-simulation",
        action="store_true",
        help="Disable room acoustics simulation",
    )
    
    # Custom advanced options
    parser.add_argument(
        "--variable-heartbeat", 
        action="store_true", 
        help="Enable variable heartbeat rate for more natural sound"
    )
    parser.add_argument(
        "--motion-smoothing", 
        action="store_true", 
        help="Enable smooth transitions between sound layers"
    )
    parser.add_argument(
        "--breathing-modulation", 
        action="store_true", 
        help="Enable maternal breathing rhythm modulation"
    )
    parser.add_argument(
        "--dynamic-volume", 
        action="store_true", 
        help="Enable automatic volume reduction over time"
    )

    # Enhanced noise generator options
    parser.add_argument(
        "--disable-equal-loudness", 
        action="store_true", 
        help="Disable equal loudness compensation"
    )
    parser.add_argument(
        "--disable-organic-drift", 
        action="store_true", 
        help="Disable organic micro-drift effect"
    )
    parser.add_argument(
        "--disable-limiter", 
        action="store_true", 
        help="Disable soft-knee limiter"
    )
    parser.add_argument(
        "--enable-diffusion", 
        action="store_true", 
        help="Enable audio diffusion polish (requires GPU)"
    )
    
    # Debug and performance options
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--seed", 
        type=int,
        help="Random seed for reproducible noise generation"
    )

    return parser.parse_args()


def display_available_options(generator):
    """Display all available options for the generator"""
    print("\n=== BabySleepSoundGenerator - Professional Audio Quality ===\n")
    print("Available predefined problem profiles:")
    for problem, desc in generator.get_problem_descriptions().items():
        print(f"  - {problem}: {desc}")

    print("\nAvailable sound types:")
    for sound in generator.get_available_sounds():
        print(f"  - {sound}")

    print("\nAvailable frequency focuses:")
    for focus in FrequencyFocus:
        print(f"  - {focus.value}: {focus.name}")

    print("\nRoom simulation options:")
    for size in RoomSize:
        print(f"  - {size.value}: {size.name} room acoustics")

    print("\nOutput format options:")
    for format in OutputFormat:
        if format == OutputFormat.WAV:
            print(f"  - {format.value}: Lossless audio, higher quality but larger file size")
        else:
            print(f"  - {format.value}: Compressed audio, smaller file size but slightly lower quality")

    from utils.optional_imports import (
        HAS_LOUDNORM, HAS_LIBROSA, HAS_PYROOMACOUSTICS,
        HAS_CUPY, HAS_TORCH, HAS_TORCHAUDIO, HAS_ISO226, HAS_AUDIO_DIFFUSION
    )
    
    print("\nFeature Status:")
    
    print("\nEBU R128 / ITU-R BS.1770 Loudness Normalization:")
    if HAS_LOUDNORM:
        print("  - Enabled: All output audio will be normalized to broadcast standards")
        print(f"  - Target loudness: {generator.target_loudness} LUFS")
    else:
        print("  - ERROR: Missing required dependency 'pyloudnorm'")

    print("\nHRTF-based Spatialization:")
    if generator.use_hrtf and HAS_LIBROSA:
        print("  - Enabled: Audio will have enhanced spatial characteristics")
    else:
        print("  - ERROR: Missing required dependency 'librosa'")

    print("\nRoom Impulse Response Simulation:")
    if generator.room_simulation and HAS_PYROOMACOUSTICS:
        print("  - Enabled: Audio will have realistic room acoustics")
    else:
        print("  - ERROR: Missing required dependency 'pyroomacoustics'")
        
    print("\nGPU Acceleration:")
    if HAS_CUPY:
        print("  - CuPy: Available for GPU-accelerated noise generation")
    else:
        print("  - ERROR: Missing required dependency 'cupy'")
        
    if HAS_TORCH:
        print("  - PyTorch: Available for neural audio processing")
    else:
        print("  - ERROR: Missing required dependency 'torch'")
        
    print("\nEnhanced Audio Features:")
    if HAS_ISO226:
        print("  - Equal Loudness Compensation: Available")
    else:
        print("  - ERROR: Missing required dependency 'iso226'")
        
    if HAS_AUDIO_DIFFUSION:
        print("  - Neural Audio Diffusion: Available for audio polish")
    else:
        print("  - ERROR: Missing required dependency 'audio_diffusion_pytorch'")
    
    print("\nAdditional setup options are available with --help")


def main():
    """Main function for command-line usage"""
    args = parse_arguments()
    
    # Configure logging level
    setup_logging(verbose=args.verbose)
    
    # Record start time for performance measurement
    start_time = time.time()

    # Create generator with specified quality settings
    generator = BabySleepSoundGenerator(
        sample_rate=args.sample_rate,
        bit_depth=args.bit_depth,
        channels=args.channels,
        target_loudness=args.target_loudness,
        use_hrtf=not args.disable_hrtf,
        room_simulation=not args.disable_room_simulation,
        seed=args.seed,
        use_equal_loudness=not args.disable_equal_loudness,
        use_limiter=not args.disable_limiter,
        use_organic_drift=not args.disable_organic_drift,
        use_diffusion=args.enable_diffusion
    )

    # Apply settings
    if args.disable_modulation:
        generator.use_dynamic_modulation = False

    if args.mode == "list":
        display_available_options(generator)

    elif args.mode == "predefined":
        if not args.problem:
            logger.error("--problem is required for predefined mode")
            return

        output_file = generator.generate_from_profile(
            problem=args.problem,
            duration_hours=args.duration,
            volume=args.volume,
            output_file=args.output,
            visualize=args.visualize,
            format=args.format,
        )

    elif args.mode == "custom":
        if not args.primary:
            logger.error("--primary is required for custom mode")
            return

        if not args.focus:
            logger.error("--focus is required for custom mode")
            return

        output_file = generator.generate_custom(
            primary_noise=args.primary,
            overlay_sounds=args.overlay or [],
            frequency_focus=FrequencyFocus(args.focus),
            duration_hours=args.duration,
            volume=args.volume or 0.6,
            spatial_width=args.spatial_width,
            room_size=RoomSize(args.room_size),
            output_file=args.output,
            visualize=args.visualize,
            format=args.format,
            variable_heartbeat=args.variable_heartbeat,
            motion_smoothing=args.motion_smoothing,
            breathing_modulation=args.breathing_modulation,
            dynamic_volume=args.dynamic_volume,
            use_equal_loudness=not args.disable_equal_loudness,
            use_limiter=not args.disable_limiter,
            use_organic_drift=not args.disable_organic_drift, 
            use_diffusion=args.enable_diffusion,
        )

    # Calculate and log total processing time
    elapsed_time = time.time() - start_time
    logger.info(f"Total processing time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()