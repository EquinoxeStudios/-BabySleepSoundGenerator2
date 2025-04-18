"""
Utility functions for Perlin noise generation.
"""

import random
import numpy as np
from utils.optional_imports import HAS_PERLIN, HAS_NUMBA

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

    Returns:
        Numpy array of noise with natural patterns
    """
    if not HAS_PERLIN:
        # Fall back to regular noise if library not available
        samples = int(duration_seconds * sample_rate)
        return np.random.normal(0, normalize_range, samples)

    samples = int(duration_seconds * sample_rate)
    result = np.zeros(samples)

    # Create seeds for each octave
    seeds = [random.randint(0, 1000) for _ in range(octaves)]

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
    if len(audio.shape) > 1:
        # Stereo audio
        return audio * modulation[:, np.newaxis]
    else:
        # Mono audio
        return audio * modulation


def generate_dynamic_modulation(
    sample_rate: int,
    duration_seconds: float,
    depth: float = 0.08,
    rate: float = 0.002,
    use_perlin: bool = True
) -> np.ndarray:
    """
    Generate a slow variation modulation curve to prevent listener fatigue.
    
    Args:
        sample_rate: Audio sample rate in Hz
        duration_seconds: Length of the audio in seconds
        depth: Modulation depth (percentage of amplitude variation)
        rate: Modulation rate in Hz (cycles per second)
        use_perlin: Whether to use Perlin noise (True) or sine wave (False)
        
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
            persistence=0.5
        )
        
        # Stretch the curve to be very slow (only a few cycles over the whole duration)
        indices = np.linspace(0, len(mod_curve) // 100, samples).astype(int)
        mod_curve = mod_curve[indices]
    else:
        # Fallback to sine wave modulation
        t = np.linspace(0, duration_seconds * rate * 2 * np.pi, samples)
        mod_curve = np.sin(t)

    # Scale to desired modulation depth
    mod_curve = 1.0 + depth * mod_curve
    
    return mod_curve