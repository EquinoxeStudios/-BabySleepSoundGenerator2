"""
Utility functions for Perlin noise generation with optimized memory usage and error handling.
"""

import logging
import numpy as np
from typing import Optional, List, Tuple, Union, Callable
import math
import functools
from scipy import signal

from utils.optional_imports import HAS_PERLIN, HAS_NUMBA
from utils.random_state import RandomStateManager
from models.constants import PerformanceConstants

logger = logging.getLogger("BabySleepSoundGenerator")

# Import optional libraries
if HAS_PERLIN:
    try:
        import noise
    except ImportError:
        HAS_PERLIN = False
        logger.warning("Failed to import 'noise' package despite it being detected")

if HAS_NUMBA:
    try:
        import numba
    except ImportError:
        HAS_NUMBA = False
        logger.warning("Failed to import 'numba' package despite it being detected")


# LRU cache for improved performance
@functools.lru_cache(maxsize=32)
def _cached_noise_kernel(scale_factor: float, octaves: int, persistence: float,
                        seed_base: int, length: int) -> np.ndarray:
    """
    Generate a cached noise kernel for reuse.
    
    Args:
        scale_factor: Scale factor for noise
        octaves: Number of octaves
        persistence: Persistence value
        seed_base: Base seed value
        length: Length of kernel
        
    Returns:
        Noise kernel array
    """
    result = np.zeros(length, dtype=np.float32)
    for j in range(octaves):
        # Each octave uses a different seed and scale
        octave_scale = scale_factor * (2**j)
        seed = seed_base + j  # Different seed per octave
        for i in range(length):
            result[i] += persistence**j * noise.pnoise1(i * octave_scale, base=seed)
    return result


def generate_perlin_noise(
    sample_rate: int,
    duration_seconds: float,
    octaves: int = 4,
    persistence: float = 0.5,
    scale_factor: float = 0.002,
    normalize: bool = True,
    normalize_range: float = 0.5,
    seed: Optional[int] = None,
    chunk_size: Optional[int] = None
) -> np.ndarray:
    """
    Generate organic noise using Perlin/Simplex noise algorithm.
    This creates more natural textures than basic random noise.

    Args:
        sample_rate: Audio sample rate in Hz
        duration_seconds: Length of the audio in seconds
        octaves: Number of layers of detail (1-8)
        persistence: How much each octave contributes to the overall shape (0.0-1.0)
        scale_factor: Controls the "speed" of changes (lower = slower variations)
        normalize: Whether to normalize the output
        normalize_range: Target normalization range (±normalize_range)
        seed: Random seed for reproducibility
        chunk_size: Optional chunk size for memory-efficient processing

    Returns:
        Numpy array of noise with natural patterns
        
    Raises:
        ValueError: For invalid parameter values
    """
    # Validate parameters
    if duration_seconds <= 0:
        raise ValueError("Duration must be positive")
    if octaves < 1 or octaves > 8:
        logger.warning(f"Invalid octaves value: {octaves}, clamping to range 1-8")
        octaves = max(1, min(8, octaves))
    if persistence < 0 or persistence > 1:
        logger.warning(f"Invalid persistence value: {persistence}, clamping to range 0-1")
        persistence = max(0, min(1, persistence))
    if scale_factor <= 0:
        logger.warning(f"Invalid scale factor: {scale_factor}, using default")
        scale_factor = 0.002
    
    # Get the random state manager with optional seed
    random_state = RandomStateManager.get_instance(seed)
    
    # Calculate total samples
    samples = int(duration_seconds * sample_rate)
    
    # Set default chunk size if not provided
    if chunk_size is None:
        chunk_size = min(samples, PerformanceConstants.MAX_DURATION_SECONDS_BEFORE_CHUNKING * sample_rate)
    
    # For very long durations, process in chunks
    if samples > chunk_size and duration_seconds > 60:
        return _generate_perlin_noise_chunked(
            sample_rate, duration_seconds, octaves, persistence, 
            scale_factor, normalize, normalize_range, seed, chunk_size
        )
    
    if not HAS_PERLIN:
        # Fall back to regular noise if library not available
        logger.info("Perlin noise library not available, falling back to normal distribution")
        return random_state.normal(0, normalize_range, samples)

    try:
        # Create a temporary result array with float32 for memory efficiency
        result = np.zeros(samples, dtype=np.float32)

        # Create seeds for each octave using the random state manager
        base_seed = seed if seed is not None else random_state.randint(0, 1000000)
        seeds = [base_seed + i for i in range(octaves)]

        # Use numba for acceleration if available
        if HAS_NUMBA:
            result = _generate_perlin_noise_numba(samples, octaves, persistence, scale_factor, seeds)
        else:
            # For better memory efficiency, process in smaller segments
            segment_size = min(samples, 10000)  # 10K samples per segment
            
            # Try to use cached kernels if possible
            use_caching = (scale_factor, octaves, persistence) == (0.002, 4, 0.5)
            
            for start in range(0, samples, segment_size):
                end = min(start + segment_size, samples)
                segment_length = end - start
                
                if use_caching:
                    # Use cached kernel for common parameters
                    kernel = _cached_noise_kernel(scale_factor, octaves, persistence, base_seed, segment_length)
                    result[start:end] = kernel
                else:
                    # Standard implementation
                    for i in range(segment_length):
                        value = 0
                        for j in range(octaves):
                            # Each octave uses a different seed and scale
                            octave_scale = scale_factor * (2**j)
                            abs_pos = start + i  # Absolute position for continuity
                            value += persistence**j * noise.pnoise1(abs_pos * octave_scale, base=seeds[j])
                        result[start + i] = value

        # Normalize if requested using vectorized operations
        if normalize and samples > 0:
            max_val = np.max(np.abs(result))
            if max_val > 0:  # Prevent division by zero
                result = normalize_range * result / max_val

        return result
    
    except Exception as e:
        logger.error(f"Error generating Perlin noise: {e}")
        # Fall back to random noise in case of error
        return random_state.normal(0, normalize_range, samples)


def _generate_perlin_noise_chunked(
    sample_rate: int,
    duration_seconds: float,
    octaves: int = 4,
    persistence: float = 0.5,
    scale_factor: float = 0.002,
    normalize: bool = True,
    normalize_range: float = 0.5,
    seed: Optional[int] = None,
    chunk_size: int = 10 * 48000  # 10 seconds at 48kHz
) -> np.ndarray:
    """
    Generate Perlin noise in chunks for very long durations with improved 
    memory efficiency and better handling of chunk boundaries.
    
    Args:
        sample_rate: Audio sample rate
        duration_seconds: Length of the sound in seconds
        octaves: Number of octave layers
        persistence: Persistence factor
        scale_factor: Scale factor
        normalize: Whether to normalize output
        normalize_range: Normalization range
        seed: Random seed
        chunk_size: Size of each processing chunk in samples
        
    Returns:
        Perlin noise array
    """
    # Calculate total samples
    total_samples = int(duration_seconds * sample_rate)
    
    # Adjust chunk size if needed
    chunk_size = min(chunk_size, PerformanceConstants.MAX_DURATION_SECONDS_BEFORE_CHUNKING * sample_rate)
    chunk_size = max(chunk_size, int(1 * sample_rate))  # At least 1 second
    
    # Calculate overlap size for smooth transitions
    crossfade_seconds = PerformanceConstants.CROSSFADE_BETWEEN_CHUNKS_SECONDS
    crossfade_samples = int(crossfade_seconds * sample_rate)
    
    # Create result array
    result = np.zeros(total_samples, dtype=np.float32)
    
    # Create seed manager to ensure consistency across chunks
    random_state = RandomStateManager.get_instance(seed)
    # Generate a base seed that will be used for all chunks
    base_seed = seed if seed is not None else random_state.randint(0, 1000000)
    
    # Calculate number of chunks
    num_chunks = (total_samples + chunk_size - crossfade_samples - 1) // (chunk_size - crossfade_samples) + 1
    
    logger.info(f"Generating Perlin noise in {num_chunks} chunks with {crossfade_samples} sample crossfade")
    
    try:
        # Process each chunk
        for chunk_idx in range(num_chunks):
            # Calculate chunk boundaries with overlap
            start_idx = chunk_idx * (chunk_size - crossfade_samples)
            end_idx = min(start_idx + chunk_size, total_samples)
            chunk_samples = end_idx - start_idx
            
            # Log progress for very long generations
            if chunk_idx % max(1, num_chunks // 10) == 0 or chunk_idx == num_chunks - 1:
                logger.info(f"Generating chunk {chunk_idx+1}/{num_chunks} "
                           f"({int(100*(chunk_idx+1)/num_chunks)}%)")
            
            # Generate this chunk with consistent seed derivation
            chunk_seed = base_seed + chunk_idx * 1000
            chunk = _generate_chunk(
                chunk_samples, octaves, persistence, scale_factor, 
                chunk_seed, start_idx, sample_rate
            )
            
            # For the first chunk, just copy directly
            if chunk_idx == 0:
                result[start_idx:end_idx] = chunk
            else:
                # For subsequent chunks, apply crossfade
                overlap_start = start_idx
                overlap_end = min(start_idx + crossfade_samples, total_samples)
                
                if overlap_end > overlap_start:
                    # Create smooth crossfade weights (cos² for equal power)
                    fade_pos = np.linspace(0, np.pi/2, overlap_end - overlap_start)
                    fade_in = np.sin(fade_pos)**2
                    fade_out = np.cos(fade_pos)**2
                    
                    # Apply crossfade in the overlap region
                    result[overlap_start:overlap_end] = (
                        result[overlap_start:overlap_end] * fade_out + 
                        chunk[:overlap_end - overlap_start] * fade_in
                    )
                    
                    # Copy the non-overlapping part
                    if overlap_end < end_idx:
                        result[overlap_end:end_idx] = chunk[overlap_end - overlap_start:
                                                          end_idx - overlap_start]
        
        # Normalize the entire result at once for consistency
        if normalize:
            max_val = np.max(np.abs(result))
            if max_val > 0:  # Prevent division by zero
                result = normalize_range * result / max_val
        
        return result
    
    except Exception as e:
        logger.error(f"Error generating chunked Perlin noise: {e}")
        # Fall back to random noise in case of error
        random_state = RandomStateManager.get_instance(seed)
        return random_state.normal(0, normalize_range, total_samples)


def _generate_chunk(
    samples: int, octaves: int, persistence: float, 
    scale_factor: float, seed: int, offset: int, sample_rate: int
) -> np.ndarray:
    """
    Generate a single chunk of Perlin noise.
    
    Args:
        samples: Number of samples to generate
        octaves: Number of octave layers
        persistence: Persistence factor
        scale_factor: Scale factor
        seed: Random seed
        offset: Sample offset for continuity
        sample_rate: Sample rate
        
    Returns:
        Noise chunk array
    """
    result = np.zeros(samples, dtype=np.float32)
    
    # Create seeds for each octave for consistency
    seeds = [seed + i for i in range(octaves)]
    
    # Use numba if available
    if HAS_NUMBA:
        temp = _generate_perlin_noise_numba(samples, octaves, persistence, scale_factor, seeds, offset)
        return temp
    
    # Standard implementation
    for i in range(samples):
        value = 0
        for j in range(octaves):
            # Each octave uses a different seed and scale
            octave_scale = scale_factor * (2**j)
            abs_pos = offset + i  # Absolute position for continuity
            value += persistence**j * noise.pnoise1(abs_pos * octave_scale, base=seeds[j])
        result[i] = value
    
    return result


def _generate_perlin_noise_numba(
    samples: int, octaves: int, persistence: float, 
    scale_factor: float, seeds: List[int], offset: int = 0
) -> np.ndarray:
    """
    Numba-accelerated Perlin noise generation for performance.
    
    Args:
        samples: Number of samples to generate
        octaves: Number of octave layers
        persistence: Persistence factor
        scale_factor: Scale factor 
        seeds: Seed list for octaves
        offset: Sample offset for continuity
        
    Returns:
        Noise array
    """
    if not HAS_NUMBA:
        logger.warning("Numba not available for Perlin noise acceleration")
        return np.zeros(samples, dtype=np.float32)
        
    try:
        # Define JIT-compiled noise approximation function
        @numba.jit(nopython=True)
        def _perlin_approx(x, seed):
            """Simple Perlin noise approximation compatible with Numba."""
            x += seed  # Add seed for variation
            
            # Integer part
            ix = int(math.floor(x))
            
            # Fractional part
            fx = x - ix
            
            # Smooth the fractional part
            u = fx * fx * (3.0 - 2.0 * fx)
            
            # Hash values
            h0 = (ix * 16807) % 2147483647
            h1 = ((ix + 1) * 16807) % 2147483647
            
            # Gradient selection and projection
            g0 = -1.0 + 2.0 * (h0 / 2147483647)
            g1 = -1.0 + 2.0 * (h1 / 2147483647)
            
            # Hermite blending
            v0 = g0 * fx
            v1 = g1 * (fx - 1.0)
            return v0 + u * (v1 - v0)

        @numba.jit(nopython=True)
        def _generate_noise(samples, octaves, persistence, scale_factor, seeds, offset):
            """JIT-compiled noise generation function."""
            result = np.zeros(samples, dtype=np.float32)
            for i in range(samples):
                value = 0.0
                for j in range(octaves):
                    # Each octave uses a different seed and scale
                    octave_scale = scale_factor * (2**j)
                    abs_pos = offset + i  # Absolute position for continuity
                    value += persistence**j * _perlin_approx(abs_pos * octave_scale, seeds[j])
                result[i] = value
            return result
            
        # Generate noise using Numba JIT compilation
        return _generate_noise(samples, octaves, persistence, scale_factor, np.array(seeds), offset)
    
    except Exception as e:
        logger.error(f"Error in Numba-accelerated Perlin noise generation: {e}")
        # Fall back to zeros in case of error with numba
        return np.zeros(samples, dtype=np.float32)


def apply_modulation(audio: np.ndarray, modulation: np.ndarray) -> np.ndarray:
    """
    Apply modulation to audio, handling both mono and stereo formats efficiently.
    
    Args:
        audio: Input audio array (mono or stereo)
        modulation: Modulation array to apply (must match audio length or be broadcastable)
        
    Returns:
        Modulated audio with same shape as input
        
    Raises:
        ValueError: If modulation shape is incompatible
    """
    try:
        # Check for empty inputs
        if audio.size == 0 or modulation.size == 0:
            return audio.copy()
        
        # Handle single-value modulation
        if modulation.size == 1:
            return audio * modulation.item()
        
        # Check length compatibility
        if len(audio) != len(modulation):
            # Try to make compatible by resizing or repeating
            if len(modulation) > len(audio):
                logger.warning(f"Modulation length ({len(modulation)}) > audio length ({len(audio)}). Truncating.")
                modulation = modulation[:len(audio)]
            else:
                # Repeat or interpolate the modulation to match length
                logger.warning(f"Modulation length ({len(modulation)}) < audio length ({len(audio)}). Resizing.")
                
                if len(modulation) * 10 < len(audio):
                    # For large differences, use interpolation
                    x_orig = np.linspace(0, 1, len(modulation))
                    x_new = np.linspace(0, 1, len(audio))
                    modulation = np.interp(x_new, x_orig, modulation)
                else:
                    # For small differences, tile the modulation
                    repeat_factor = int(np.ceil(len(audio) / len(modulation)))
                    modulation = np.tile(modulation, repeat_factor)[:len(audio)]
        
        # Check if modulation needs to be expanded for stereo
        if len(audio.shape) > 1 and len(modulation.shape) == 1:
            # Use broadcasting for efficiency
            return audio * modulation.reshape(-1, 1)
        else:
            # Direct multiplication for mono or already expanded modulation
            return audio * modulation
            
    except Exception as e:
        logger.error(f"Error applying modulation: {e}")
        # Return original audio if modulation fails
        return audio.copy()


def generate_dynamic_modulation(
    sample_rate: int,
    duration_seconds: float,
    depth: float = 0.08,
    rate: float = 0.002,
    use_perlin: bool = True,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate a slow variation modulation curve to prevent listener fatigue.
    
    Args:
        sample_rate: Audio sample rate in Hz
        duration_seconds: Length of the audio in seconds
        depth: Modulation depth (percentage of amplitude variation) [0.0-1.0]
        rate: Modulation rate in Hz (cycles per second) [0.0001-0.1]
        use_perlin: Whether to use Perlin noise (True) or sine wave (False)
        seed: Random seed for reproducibility
        
    Returns:
        Modulation curve as numpy array
        
    Raises:
        ValueError: For invalid parameter values
    """
    # Validate parameters
    if duration_seconds <= 0:
        raise ValueError("Duration must be positive")
    if depth < 0 or depth > 1:
        logger.warning(f"Invalid modulation depth: {depth}, clamping to range 0-1")
        depth = max(0, min(1, depth))
    if rate < 0.0001 or rate > 0.1:
        logger.warning(f"Invalid modulation rate: {rate}, clamping to range 0.0001-0.1")
        rate = max(0.0001, min(0.1, rate))
    
    samples = int(duration_seconds * sample_rate)
    
    try:
        # Create very slow modulation curve
        if HAS_PERLIN and use_perlin:
            # Use perlin noise for organic modulation
            # Generate a shorter noise and stretch it for better performance
            perlin_duration = min(duration_seconds, 60.0)  # Cap at 60 seconds for memory efficiency
            mod_curve = generate_perlin_noise(
                sample_rate, 
                perlin_duration, 
                octaves=1, 
                persistence=0.5,
                seed=seed
            )
            
            # For very long durations, stretch the curve
            if perlin_duration < duration_seconds:
                # Create indices safely to avoid out of bounds
                stretch_factor = PerformanceConstants.PERLIN_STRETCH_FACTOR
                indices = np.linspace(0, len(mod_curve) - 1, samples)
                indices = np.clip(indices.astype(int), 0, len(mod_curve) - 1)
                mod_curve = mod_curve[indices]
            
            # Apply a low-pass filter for smoother variations
            cutoff = min(rate * 4, 0.4 * sample_rate / 2)
            b, a = signal.butter(2, cutoff / (sample_rate / 2), 'low')
            mod_curve = signal.filtfilt(b, a, mod_curve)
        else:
            # Generate sine wave modulation
            logger.info("Using sine wave modulation instead of Perlin noise")
            t = np.linspace(0, duration_seconds * rate * 2 * np.pi, samples)
            mod_curve = np.sin(t)
            
            # Add a slight variation to the sine wave
            if seed is not None:
                random_state = RandomStateManager.get_instance(seed)
                variation = 0.2 * random_state.normal(0, 0.05, len(mod_curve))
                # Apply low-pass filter to the variation for smoothness
                b, a = signal.butter(2, 0.01, 'low')
                variation = signal.filtfilt(b, a, variation)
                mod_curve += variation

        # Scale to desired modulation depth
        mod_curve = 1.0 + depth * np.clip(mod_curve, -1.0, 1.0)
        
        return mod_curve
        
    except Exception as e:
        logger.error(f"Error generating dynamic modulation: {e}")
        # Return a constant modulation (no effect) if generation fails
        return np.ones(samples, dtype=np.float32)