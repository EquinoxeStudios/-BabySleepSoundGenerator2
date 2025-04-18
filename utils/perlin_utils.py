"""
Utility functions for Perlin noise generation.
"""

import random
import numpy as np
from utils.optional_imports import HAS_PERLIN, HAS_NUMBA
from utils.random_state import RandomStateManager
from models.constants import PerformanceConstants

# Import optional libraries
if HAS_PERLIN:
    import noise

if HAS_NUMBA:
    import numba


def generate_perlin_noise(
    sample_rate: int,
    duration_seconds: float,
    octaves: int = 4,
    persistence: float = 0.5,
    scale_factor: float = 0.002,
    normalize: bool = True,
    normalize_range: float = 0.5,
    seed: int = None,
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
        return random_state.normal(0, normalize_range, samples)

    result = np.zeros(samples)

    # Create seeds for each octave using the random state manager
    seeds = [random_state.randint(0, 1000) for _ in range(octaves)]

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


def _generate_perlin_noise_chunked(
    sample_rate: int,
    duration_seconds: float,
    octaves: int = 4,
    persistence: float = 0.5,
    scale_factor: float = 0.002,
    normalize: bool = True,
    normalize_range: float = 0.5,
    seed: int = None,
) -> np.ndarray:
    """Generate Perlin noise in chunks for very long durations."""
    # Calculate total samples
    total_samples = int(duration_seconds * sample_rate)
    
    # Process in chunks
    chunk_seconds = PerformanceConstants.FFT_CHUNK_SIZE_SECONDS
    chunk_samples = int(chunk_seconds * sample_rate)
    
    # Calculate number of chunks
    num_chunks = (total_samples + chunk_samples - 1) // chunk_samples
    
    # Create result array
    result = np.zeros(total_samples)
    
    # Create seed manager to ensure consistency across chunks
    random_state = RandomStateManager.get_instance(seed)
    chunk_seed = seed if seed is not None else random_state.randint(0, 1000000)
    
    # Process each chunk
    for i in range(num_chunks):
        # Calculate chunk boundaries
        start_idx = i * chunk_samples
        end_idx = min(start_idx + chunk_samples, total_samples)
        
        # Generate this chunk with consistent seeding
        chunk_random_state = RandomStateManager.get_instance(chunk_seed + i)
        chunk_seeds = [chunk_random_state.randint(0, 1000) for _ in range(octaves)]
        
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
        
        # Store in result array
        result[start_idx:end_idx] = chunk
    
    # Normalize the entire result at once for consistency
    if normalize:
        max_val = np.max(np.abs(result))
        if max_val > 0:  # Prevent division by zero
            result = normalize_range * result / max_val
    
    return result.astype(np.float32)


def _generate_perlin_noise_numba(samples, octaves, persistence, scale_factor, seeds):
    """Numba-accelerated Perlin noise generation if available"""
    if not HAS_NUMBA:
        return np.zeros(samples)
        
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


def apply_modulation(audio: np.ndarray, modulation: np.ndarray) -> np.ndarray:
    """
    Apply modulation to audio, handling both mono and stereo formats.
    
    Args:
        audio: Input audio array (mono or stereo)
        modulation: Modulation array to apply (must match audio length)
        
    Returns:
        Modulated audio with same shape as input
    """
    # Check if modulation needs to be expanded for stereo
    if len(audio.shape) > 1 and len(modulation.shape) == 1:
        # Prepare for broadcasting by adding dimension
        return audio * modulation[:, np.newaxis]
    else:
        # Direct multiplication for mono or already expanded modulation
        return audio * modulation


def generate_dynamic_modulation(
    sample_rate: int,
    duration_seconds: float,
    depth: float = 0.08,
    rate: float = 0.002,
    use_perlin: bool = True,
    seed: int = None
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
        indices = np.linspace(0, len(mod_curve) // stretch_factor, samples).astype(int)
        mod_curve = mod_curve[np.clip(indices, 0, len(mod_curve) - 1)]
    else:
        # Fallback to sine wave modulation
        t = np.linspace(0, duration_seconds * rate * 2 * np.pi, samples)
        mod_curve = np.sin(t)

    # Scale to desired modulation depth
    mod_curve = 1.0 + depth * mod_curve
    
    return mod_curve