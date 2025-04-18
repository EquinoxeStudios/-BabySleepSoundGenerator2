"""
Utility functions for Perlin noise generation.
"""

import logging
import numpy as np
from typing import Optional, List, Tuple

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


def generate_perlin_noise(
    sample_rate: int,
    duration_seconds: float,
    octaves: int = 4,
    persistence: float = 0.5,
    scale_factor: float = 0.002,
    normalize: bool = True,
    normalize_range: float = 0.5,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate organic noise using Perlin/Simplex noise algorithm.
    This creates more natural textures than basic random noise.

    Args:
        sample_rate: Audio sample rate in Hz
        duration_seconds: Length of the audio in seconds
        octaves: Number of layers of detail
        persistence: How much each octave contributes to the overall shape
        scale_factor: Controls the "speed" of changes (lower = slower variations)
        normalize: Whether to normalize the output
        normalize_range: Target normalization range (±normalize_range)
        seed: Random seed for reproducibility

    Returns:
        Numpy array of noise with natural patterns
    """
    # Get the random state manager with optional seed
    random_state = RandomStateManager.get_instance(seed)
    
    # Calculate total samples
    samples = int(duration_seconds * sample_rate)
    
    # For very long durations, process in chunks
    if samples > PerformanceConstants.MAX_DURATION_SECONDS_BEFORE_CHUNKING * sample_rate and duration_seconds > 60:
        return _generate_perlin_noise_chunked(
            sample_rate, duration_seconds, octaves, persistence, 
            scale_factor, normalize, normalize_range, seed
        )
    
    if not HAS_PERLIN:
        # Fall back to regular noise if library not available
        logger.info("Perlin noise library not available, falling back to normal distribution")
        return random_state.normal(0, normalize_range, samples)

    try:
        result = np.zeros(samples)

        # Create seeds for each octave using the random state manager
        base_seed = seed if seed is not None else random_state.randint(0, 1000000)
        seeds = [base_seed + i for i in range(octaves)]

        # Use numba for acceleration if available
        if HAS_NUMBA:
            result = _generate_perlin_noise_numba(samples, octaves, persistence, scale_factor, seeds)
        else:
            # Standard implementation
            for i in range(samples):
                value = 0
                for j in range(octaves):
                    # Each octave uses a different seed and scale
                    octave_scale = scale_factor * (2**j)
                    value += persistence**j * noise.pnoise1(i * octave_scale, base=seeds[j])

                result[i] = value

        # Normalize if requested
        if normalize:
            max_val = np.max(np.abs(result))
            if max_val > 0:  # Prevent division by zero
                result = normalize_range * result / max_val

        return result.astype(np.float32)
    
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
) -> np.ndarray:
    """Generate Perlin noise in chunks for very long durations."""
    # Calculate total samples
    total_samples = int(duration_seconds * sample_rate)
    
    # Process in chunks
    chunk_seconds = PerformanceConstants.FFT_CHUNK_SIZE_SECONDS
    chunk_samples = int(chunk_seconds * sample_rate)
    crossfade_samples = int(PerformanceConstants.CROSSFADE_BETWEEN_CHUNKS_SECONDS * sample_rate)
    
    # Calculate number of chunks
    num_chunks = (total_samples + chunk_samples - 1) // chunk_samples
    
    # Create result array
    result = np.zeros(total_samples)
    
    # Create seed manager to ensure consistency across chunks
    random_state = RandomStateManager.get_instance(seed)
    # Generate a base seed that will be used for all chunks
    base_seed = seed if seed is not None else random_state.randint(0, 1000000)
    
    logger.info(f"Generating Perlin noise in {num_chunks} chunks of {chunk_seconds}s each")
    
    try:
        # Process each chunk
        for i in range(num_chunks):
            # Calculate chunk boundaries
            start_idx = i * chunk_samples
            end_idx = min(start_idx + chunk_samples + crossfade_samples, total_samples)
            chunk_duration = (end_idx - start_idx) / sample_rate
            
            # Generate this chunk with consistent seeding
            # Use consistent seed derivation based on the chunk index
            chunk_seed = base_seed + i * 1000
            chunk_seeds = [chunk_seed + j for j in range(octaves)]
            
            # Log progress for very long generations
            if i % max(1, num_chunks // 10) == 0 or i == num_chunks - 1:
                logger.info(f"Generating chunk {i+1}/{num_chunks} ({int(100*(i+1)/num_chunks)}%)")
            
            if HAS_NUMBA:
                chunk = _generate_perlin_noise_numba(
                    end_idx - start_idx, octaves, persistence, scale_factor, chunk_seeds)
            else:
                chunk = np.zeros(end_idx - start_idx)
                for j in range(end_idx - start_idx):
                    value = 0
                    for k in range(octaves):
                        # Use consistent indexing across chunks
                        abs_idx = start_idx + j
                        octave_scale = scale_factor * (2**k)
                        value += persistence**k * noise.pnoise1(abs_idx * octave_scale, base=chunk_seeds[k])
                    chunk[j] = value
            
            # Apply crossfade if not the first chunk
            if i > 0:
                # Create overlap region with previous chunk
                overlap_start = start_idx
                overlap_end = min(start_idx + crossfade_samples, total_samples)
                
                if overlap_end > overlap_start:
                    # Create crossfade weights
                    fade_in = np.linspace(0, 1, overlap_end - overlap_start)
                    fade_out = 1 - fade_in
                    
                    # Apply crossfade in the overlap region
                    result[overlap_start:overlap_end] = (
                        result[overlap_start:overlap_end] * fade_out + 
                        chunk[:overlap_end - overlap_start] * fade_in
                    )
                    
                    # Copy the non-overlapping part
                    if overlap_end < end_idx:
                        result[overlap_end:end_idx] = chunk[overlap_end - overlap_start:end_idx - overlap_start]
            else:
                # First chunk, no crossfade needed
                result[start_idx:end_idx] = chunk[:end_idx - start_idx]
        
        # Normalize the entire result at once for consistency
        if normalize:
            max_val = np.max(np.abs(result))
            if max_val > 0:  # Prevent division by zero
                result = normalize_range * result / max_val
        
        return result.astype(np.float32)
    
    except Exception as e:
        logger.error(f"Error generating chunked Perlin noise: {e}")
        # Fall back to random noise in case of error
        random_state = RandomStateManager.get_instance(seed)
        return random_state.normal(0, normalize_range, total_samples)


def _generate_perlin_noise_numba(samples: int, octaves: int, persistence: float, 
                               scale_factor: float, seeds: List[int]) -> np.ndarray:
    """Numba-accelerated Perlin noise generation if available"""
    if not HAS_NUMBA:
        logger.warning("Numba not available for Perlin noise acceleration")
        return np.zeros(samples)
        
    try:
        @numba.jit(nopython=True)
        def _generate_noise(samples, octaves, persistence, scale_factor, seeds):
            result = np.zeros(samples)
            for i in range(samples):
                value = 0.0
                for j in range(octaves):
                    # Each octave uses a different seed and scale
                    octave_scale = scale_factor * (2**j)
                    # Simplified noise approximation for numba compatibility
                    x = i * octave_scale + seeds[j]
                    n = (np.sin(x) * 43758.5453) % 1
                    value += persistence**j * n
                result[i] = value
            return result
            
        return _generate_noise(samples, octaves, persistence, scale_factor, np.array(seeds))
    
    except Exception as e:
        logger.error(f"Error in Numba-accelerated Perlin noise generation: {e}")
        # Fall back to zeros in case of error with numba
        return np.zeros(samples)


def apply_modulation(audio: np.ndarray, modulation: np.ndarray) -> np.ndarray:
    """
    Apply modulation to audio, handling both mono and stereo formats.
    
    Args:
        audio: Input audio array (mono or stereo)
        modulation: Modulation array to apply (must match audio length)
        
    Returns:
        Modulated audio with same shape as input
    """
    try:
        # Validate input dimensions
        if len(audio) != len(modulation) and len(modulation) != 1:
            logger.warning(f"Modulation length ({len(modulation)}) doesn't match audio length ({len(audio)})")
            # Resize modulation array if possible
            if len(modulation) > len(audio):
                modulation = modulation[:len(audio)]
            else:
                # Repeat the modulation to match length
                repeat_factor = int(np.ceil(len(audio) / len(modulation)))
                modulation = np.tile(modulation, repeat_factor)[:len(audio)]
        
        # Check if modulation needs to be expanded for stereo
        if len(audio.shape) > 1 and len(modulation.shape) == 1:
            # Prepare for broadcasting by adding dimension
            return audio * modulation[:, np.newaxis]
        else:
            # Direct multiplication for mono or already expanded modulation
            return audio * modulation
            
    except Exception as e:
        logger.error(f"Error applying modulation: {e}")
        # Return original audio if modulation fails
        return audio


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
        depth: Modulation depth (percentage of amplitude variation)
        rate: Modulation rate in Hz (cycles per second)
        use_perlin: Whether to use Perlin noise (True) or sine wave (False)
        seed: Random seed for reproducibility
        
    Returns:
        Modulation curve as numpy array
    """
    samples = int(duration_seconds * sample_rate)
    
    try:
        # Create very slow modulation curve
        if HAS_PERLIN and use_perlin:
            # Use perlin noise for organic modulation
            mod_curve = generate_perlin_noise(
                sample_rate, 
                duration_seconds, 
                octaves=1, 
                persistence=0.5,
                seed=seed
            )
            
            # Stretch the curve to be very slow (only a few cycles over the whole duration)
            stretch_factor = PerformanceConstants.PERLIN_STRETCH_FACTOR
            
            # Create indices safely to avoid out of bounds
            indices = np.linspace(0, len(mod_curve) // stretch_factor - 1, samples)
            indices = np.clip(indices.astype(int), 0, len(mod_curve) - 1)
            
            mod_curve = mod_curve[indices]
        else:
            # Fallback to sine wave modulation
            logger.info("Using sine wave modulation instead of Perlin noise")
            t = np.linspace(0, duration_seconds * rate * 2 * np.pi, samples)
            mod_curve = np.sin(t)

        # Scale to desired modulation depth
        mod_curve = 1.0 + depth * mod_curve
        
        return mod_curve
        
    except Exception as e:
        logger.error(f"Error generating dynamic modulation: {e}")
        # Return a constant modulation (no effect) if generation fails
        return np.ones(samples)