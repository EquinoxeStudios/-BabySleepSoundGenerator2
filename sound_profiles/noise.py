"""
Enhanced noise generator with streaming, GPU support, and white/pink/brown flavours.
"""

import logging
import os
import subprocess
import tempfile
import numpy as np
from scipy import signal
from typing import Optional, Dict, Any, Union, Tuple, List, Callable
import threading
from functools import lru_cache

from sound_profiles.base import SoundProfileGenerator
from utils.optional_imports import HAS_PERLIN
from utils.perlin_utils import generate_perlin_noise, apply_modulation, generate_dynamic_modulation
from utils.random_state import RandomStateManager
from models.constants import PerformanceConstants, Constants
from models.enums import NoiseColor

logger = logging.getLogger("BabySleepSoundGenerator")

# Import optional dependencies - lazy loaded when needed
_HAS_CUPY = False
_HAS_TORCH = False 
_HAS_TORCHAUDIO = False
_HAS_ISO226 = False
_HAS_AUDIO_DIFFUSION = False
_SOUNDFILE_AVAILABLE = False

# Globals for cached resources
_CACHED_FILTERS = {}
_CACHED_EQUAL_LOUDNESS = {}
_CACHED_DIFFUSION_MODEL = None
_FILTER_CACHE_LOCK = threading.Lock()
_DIFFUSION_CACHE_LOCK = threading.Lock()
_RNG_CACHE: Dict[int, Tuple[Any, bool]] = {}
_RNG_CACHE_LOCK = threading.Lock()


def _try_import_dependencies():
    """Lazy-load optional dependencies"""
    global _HAS_CUPY, _HAS_TORCH, _HAS_TORCHAUDIO, _HAS_ISO226, _HAS_AUDIO_DIFFUSION, _SOUNDFILE_AVAILABLE
    
    # Only attempt imports if not already checked
    if not any([_HAS_CUPY, _HAS_TORCH, _HAS_TORCHAUDIO, _HAS_ISO226, _HAS_AUDIO_DIFFUSION]):
        try:
            import cupy
            _HAS_CUPY = True
            logger.info("CuPy found - GPU acceleration available")
        except ImportError:
            logger.info("CuPy not found - using CPU fallback")
            _HAS_CUPY = False
            
        try:
            import torch
            _HAS_TORCH = True
            logger.info("PyTorch found")
        except ImportError:
            logger.info("PyTorch not found - some features will be limited")
            _HAS_TORCH = False
            
        try:
            import torchaudio
            import torchaudio.prototype.functional as F
            _HAS_TORCHAUDIO = True
            logger.info(f"TorchAudio found: version {torchaudio.__version__}")
        except ImportError:
            logger.info("TorchAudio not found - using scipy fallback for filters")
            _HAS_TORCHAUDIO = False
            
        try:
            import iso226
            _HAS_ISO226 = True
            logger.info("ISO-226 library found")
        except ImportError:
            logger.info("ISO-226 library not found - equal loudness compensation will be limited")
            _HAS_ISO226 = False
            
        try:
            import audio_diffusion_pytorch
            _HAS_AUDIO_DIFFUSION = True
            logger.info("Audio Diffusion library found")
        except ImportError:
            logger.info("Audio Diffusion library not found - diffusion polish unavailable")
            _HAS_AUDIO_DIFFUSION = False
            
        try:
            import soundfile
            _SOUNDFILE_AVAILABLE = True
        except ImportError:
            logger.info("SoundFile not found - export functionality will be limited")
            _SOUNDFILE_AVAILABLE = False


def _get_thread_safe_rng(seed=None):
    """Get a thread-local RNG to avoid thread safety issues"""
    thread_id = threading.get_ident()
    
    with _RNG_CACHE_LOCK:
        if thread_id not in _RNG_CACHE:
            rng, is_gpu = _get_high_quality_rng(seed)
            _RNG_CACHE[thread_id] = (rng, is_gpu)
    
    return _RNG_CACHE[thread_id]


def _get_high_quality_rng(seed=None):
    """Get high-quality random number generator with PCG64DXSM"""
    # Try to get random seed from os.urandom if not provided
    if seed is None:
        try:
            # Get 8 bytes (64 bits) of random data
            random_bytes = os.urandom(8)
            seed = int.from_bytes(random_bytes, byteorder='little')
            logger.debug(f"Using OS-generated random seed: {seed}")
        except (OSError, NotImplementedError):
            # Fallback to RandomStateManager if os.urandom is not available
            state_manager = RandomStateManager.get_instance()
            seed = state_manager.randint(0, 2**63-1)
            logger.debug(f"Using fallback random seed: {seed}")
    
    # Try to use CuPy's PCG64DXSM if available
    global _HAS_CUPY
    if not _HAS_CUPY:
        _try_import_dependencies()
        
    if _HAS_CUPY:
        try:
            import cupy
            # Try to initialize CuPy's PCG64DXSM RNG
            try:
                rng = cupy.random.Generator(cupy.random.PCG64DXSM(seed))
                # Test if it works by allocating a small array
                cupy.asnumpy(rng.normal(0, 1, 10))
                return rng, True  # Return RNG and GPU flag
            except (TypeError, AttributeError, cupy.cuda.runtime.CUDARuntimeError):
                # Older CuPy versions may not have PCG64DXSM or CUDA not available
                logger.info("CuPy found but PCG64DXSM or CUDA not available - falling back to NumPy")
        except Exception as e:
            logger.warning(f"Error initializing CuPy RNG: {e}")
    
    # Fallback to NumPy PCG64DXSM
    try:
        from numpy.random import Generator, PCG64DXSM
        rng = Generator(PCG64DXSM(seed))
        return rng, False  # Return RNG and GPU flag
    except (ImportError, AttributeError):
        # Fallback to RandomStateManager if PCG64DXSM is not available
        logger.warning("PCG64DXSM not available - using RandomStateManager")
        state_manager = RandomStateManager.get_instance(seed)
        return state_manager, False


@lru_cache(maxsize=8)
def _design_dc_alias_filters(sample_rate, hp_cutoff=20.0, lp_cutoff=18000.0):
    """Design DC removal and anti-aliasing filters"""
    global _HAS_TORCHAUDIO
    if not _HAS_TORCHAUDIO:
        _try_import_dependencies()
    
    if _HAS_TORCHAUDIO and _HAS_TORCH:
        import torch
        try:
            import torchaudio
            # First try the newer path (TorchAudio v2)
            try:
                import torchaudio.functional as F
                logger.info("Using torchaudio.functional API (v2+)")
            except ImportError:
                # Fall back to prototype API (pre-v2)
                import torchaudio.prototype.functional as F
                logger.info("Using torchaudio.prototype.functional API (pre-v2)")
            
            # Check TorchAudio version for parameter compatibility
            # For TorchAudio >= 0.13, use cutoff_freq
            # For earlier versions, might need cutoff instead
            
            try:
                # Design highpass filter for DC removal
                hp_b, hp_a = F.design_biquad(
                    sample_rate=sample_rate,
                    btype='highpass',
                    cutoff_freq=hp_cutoff,
                    Q=0.707
                )
                
                # Design lowpass filter for anti-aliasing
                lp_b, lp_a = F.design_biquad(
                    sample_rate=sample_rate,
                    btype='lowpass',
                    cutoff_freq=min(lp_cutoff, sample_rate/2 - 1000),  # Ensure below Nyquist
                    Q=0.707
                )
            except TypeError as e:
                logger.warning(f"Parameter error with TorchAudio functional.design_biquad: {e}")
                logger.info("Trying alternative parameter name 'cutoff' for TorchAudio < 0.13")
                
                # Try with alternative parameter name for older versions
                hp_b, hp_a = F.design_biquad(
                    sample_rate=sample_rate,
                    btype='highpass',
                    cutoff=hp_cutoff,
                    Q=0.707
                )
                
                lp_b, lp_a = F.design_biquad(
                    sample_rate=sample_rate,
                    btype='lowpass',
                    cutoff=min(lp_cutoff, sample_rate/2 - 1000),  # Ensure below Nyquist
                    Q=0.707
                )
        except ImportError as e:
            logger.warning(f"Error importing TorchAudio modules: {e}")
            logger.info("Falling back to SciPy for filter design")
            # Fall through to SciPy fallback
        
        # Convert to numpy arrays
        hp_b = hp_b.numpy()
        hp_a = hp_a.numpy()
        lp_b = lp_b.numpy()
        lp_a = lp_a.numpy()
    else:
        # Fallback to scipy for filter design
        # Highpass filter coefficients (2-pole Butterworth)
        hp_b, hp_a = signal.butter(2, hp_cutoff / (sample_rate / 2), 'high')
        
        # Lowpass filter coefficients (2-pole Butterworth)
        lp_b, lp_a = signal.butter(2, min(lp_cutoff, sample_rate/2 - 1000) / (sample_rate / 2), 'low')
    
    return (hp_b, hp_a), (lp_b, lp_a)


def _get_diffusion_model():
    """Get the diffusion model with thread-safe caching and optional disabling"""
    global _CACHED_DIFFUSION_MODEL, _HAS_AUDIO_DIFFUSION
    
    # Check for environment flag to disable diffusion in production
    if os.environ.get("DISABLE_AUDIO_DIFFUSION") == "1":
        logger.info("Audio diffusion disabled by environment variable")
        return None
    
    if not _HAS_AUDIO_DIFFUSION:
        _try_import_dependencies()
        if not _HAS_AUDIO_DIFFUSION:
            return None
        
    with _DIFFUSION_CACHE_LOCK:
        if _CACHED_DIFFUSION_MODEL is None:
            try:
                import torch
                from audio_diffusion_pytorch import DiffusionModel, UNetV0
                
                # Check for CUDA availability to avoid GPU OOM issues
                device = "cuda" if torch.cuda.is_available() else "cpu"
                if device == "cpu":
                    logger.warning("CUDA unavailable for diffusion model - using CPU (slow)")
                
                logger.info(f"Loading audio diffusion model on {device}")
                _CACHED_DIFFUSION_MODEL = DiffusionModel(net_t=UNetV0).to(device)
                
                # Free excess memory after model load
                if device == "cuda":
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                logger.error(f"Failed to load diffusion model: {e}")
                return None
    
    return _CACHED_DIFFUSION_MODEL


@lru_cache(maxsize=4)
def _get_equal_loudness_filter(sample_rate, level=50):
    """Get ISO-226 equal-loudness filter FIR taps"""
    global _HAS_ISO226
    if not _HAS_ISO226:
        _try_import_dependencies()
    
    if _HAS_ISO226:
        try:
            import iso226
            # Get equal loudness filter taps
            eq_data = iso226.equal_loudness_weight(fs=sample_rate, level=level)
            fir_taps = eq_data['FIR']
            
            # Make sure the taps are odd length for linear phase
            if len(fir_taps) % 2 == 0:
                fir_taps = np.append(fir_taps, 0)
            
            # Normalize to unity gain at DC to maintain proper levels
            fir_taps = fir_taps / np.sum(fir_taps)
            
            return fir_taps
        except Exception as e:
            logger.warning(f"Error creating ISO-226 filter: {e}")
    
    # Fallback: approximate A-weighting filter
    logger.info("Using approximate A-weighting as equal-loudness fallback")
    # Create an approximate A-weighting filter (simplistic version)
    nyquist = sample_rate / 2
    n_taps = min(int(sample_rate * 0.05), 1001)  # 50ms or 1001 taps, whichever is smaller
    if n_taps % 2 == 0:
        n_taps += 1  # Ensure odd length
        
    # Create frequency points from 0 to Nyquist
    freqs = np.linspace(0, nyquist, n_taps // 2 + 1)
    
    # Create A-weighting response
    f_sq = np.square(freqs)
    numerator = 12200**2 * f_sq**2
    denominator = (f_sq + 20.6**2) * np.sqrt((f_sq + 107.7**2) * (f_sq + 737.9**2)) * (f_sq + 12200**2)
    a_weighting = 2.0 + 20 * np.log10(numerator / denominator + 1e-10)
    
    # Convert to linear scale
    a_weighting = 10 ** (a_weighting / 20)
    
    # Set DC component to very small value
    a_weighting[0] = 0.001
    
    # Create symmetric frequency response for a linear-phase filter
    full_response = np.concatenate([a_weighting, a_weighting[-2:0:-1]])
    
    # Convert to time domain using IFFT
    fir_taps = np.real(np.fft.ifft(full_response))
    
    # Shift to make causal and apply window
    fir_taps = np.roll(fir_taps, n_taps // 2)
    window = np.hamming(n_taps)
    fir_taps = fir_taps[:n_taps] * window
    
    # Normalize to unity gain to maintain proper levels
    fir_taps = fir_taps / np.sum(fir_taps)
    
    return fir_taps


@lru_cache(maxsize=4)
def _design_crossover_filters(sample_rate, num_bands=10, min_freq=20, max_freq=20000):
    """Design linear-phase FIR crossover filters for half-octave bands"""
    # Calculate band frequencies (logarithmic spacing)
    band_freqs = np.logspace(np.log10(min_freq), np.log10(max_freq), num_bands + 1)
    
    # Filter length - for good crossover performance
    # Longer is better but more CPU intensive
    filter_length = min(int(sample_rate * 0.05), 1001)  # 50ms or 1001 taps, whichever is smaller
    if filter_length % 2 == 0:
        filter_length += 1  # Ensure odd length for linear phase
    
    filters = []
    
    # Check if we have a cached result on disk
    import os
    import pickle
    
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "babysleepsound")
    os.makedirs(cache_dir, exist_ok=True)
    
    cache_file = os.path.join(cache_dir, f"crossover_filters_{sample_rate}_{num_bands}_{min_freq}_{max_freq}.pkl")
    
    # Try to load from cache
    try:
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
    except Exception as e:
        logger.warning(f"Failed to load crossover filters from cache: {e}")
    
    # If not cached or loading failed, compute filters
    for i in range(num_bands):
        # Band edges
        low_freq = band_freqs[i]
        high_freq = band_freqs[i + 1]
        
        # Normalize to Nyquist
        nyquist = sample_rate / 2
        low_norm = low_freq / nyquist
        high_norm = min(high_freq / nyquist, 0.99)  # Ensure below Nyquist
        
        # Design bandpass filter
        if i == 0:
            # Lowpass for first band
            b = signal.firwin(filter_length, high_norm, window='hamming')
        elif i == num_bands - 1:
            # Highpass for last band
            b = signal.firwin(filter_length, low_norm, pass_zero=False, window='hamming')
        else:
            # Bandpass for middle bands
            b = signal.firwin(filter_length, [low_norm, high_norm], pass_zero=False, window='hamming')
        
        filters.append(b)
    
    result = (filters, band_freqs)
    
    # Save to cache
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
    except Exception as e:
        logger.warning(f"Failed to save crossover filters to cache: {e}")
    
    return result


def _apply_organic_drift(audio, sample_rate, use_perlin=True, seed=None):
    """Apply organic micro-drift using half-octave band modulation"""
    # Get or design crossover filters
    with _FILTER_CACHE_LOCK:
        cache_key = f"crossover_{sample_rate}"
        if cache_key not in _CACHED_FILTERS:
            filters, band_freqs = _design_crossover_filters(sample_rate)
            _CACHED_FILTERS[cache_key] = (filters, band_freqs)
        else:
            filters, band_freqs = _CACHED_FILTERS[cache_key]
    
    # Initialize random state
    random_state = RandomStateManager.get_instance(seed)
    
    # Frequency-domain modulation for performance
    # Take FFT of the full signal once
    X = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(len(audio), 1/sample_rate)
    
    # Create band-filtered signals directly in frequency domain
    bands = []
    for i, filt in enumerate(filters):
        # Compute the frequency response of the filter
        w, h = signal.freqz(filt, 1.0, worN=len(freqs), fs=sample_rate)
        
        # Interpolate to match our frequency points if needed
        if len(w) != len(freqs):
            h = np.interp(freqs, w, np.abs(h))
        else:
            h = np.abs(h)
            
        # Apply the filter in frequency domain
        X_band = X.copy() * h
        
        # Convert back to time domain
        band = np.fft.irfft(X_band, len(audio))
        bands.append(band)
    
    # Generate LFO modulators for each band
    duration_seconds = len(audio) / sample_rate
    modulators = []
    
    for i in range(len(bands)):
        # Random LFO period between 5-20 seconds
        period_seconds = random_state.uniform(5.0, 20.0)
        
        if use_perlin and HAS_PERLIN:
            # Generate Perlin noise for organic modulation
            modulator = generate_perlin_noise(
                sample_rate, 
                duration_seconds, 
                octaves=1, 
                persistence=0.5,
                scale_factor=1.0/(period_seconds*10),  # Adjust scale for desired period
                seed=seed + i if seed is not None else None  # Different seed per band
            )
        else:
            # Fallback to sine LFO
            t = np.linspace(0, duration_seconds, len(audio))
            # Random phase offset
            phase_offset = random_state.uniform(0, 2*np.pi)
            modulator = np.sin(2 * np.pi * (1.0/period_seconds) * t + phase_offset)
        
        # Scale to ±1 dB (convert from dB to linear)
        db_variation = 1.0
        modulator = np.power(10, modulator * db_variation / 20)
        modulators.append(modulator)
    
    # Apply modulators to each band
    for i in range(len(bands)):
        bands[i] = bands[i] * modulators[i]
    
    # Sum all bands back together
    modulated_audio = np.sum(bands, axis=0)
    
    # Normalize to prevent clipping
    max_val = np.max(np.abs(modulated_audio))
    if max_val > Constants.MAX_AUDIO_VALUE:
        modulated_audio = modulated_audio / max_val * Constants.MAX_AUDIO_VALUE
    
    return modulated_audio


def _apply_soft_knee_limiter(audio, sample_rate, threshold_db=-10, knee_db=6, attack_ms=5, release_ms=50, lookahead_ms=10):
    """Apply a soft-knee A-weighted limiter"""
    # Convert parameters to samples (ensure at least 1 sample)
    attack_samples = max(1, int(attack_ms * sample_rate / 1000))
    release_samples = max(1, int(release_ms * sample_rate / 1000))
    lookahead_samples = max(1, int(lookahead_ms * sample_rate / 1000))
    
    # Convert threshold from dB to linear
    threshold_linear = np.power(10, threshold_db / 20)
    knee_range = np.power(10, knee_db / 20)
    
    # Pad the beginning for lookahead
    padded_audio = np.pad(audio, (lookahead_samples, 0), mode='edge')
    
    # Initialize gain reduction array
    gain_reduction = np.ones(len(padded_audio))
    
    # Calculate A-weighting filter if needed
    global _HAS_ISO226
    if not _HAS_ISO226:
        _try_import_dependencies()
    
    # Get or compute A-weighting filter taps
    with _FILTER_CACHE_LOCK:
        cache_key = f"a_weight_{sample_rate}"
        if cache_key not in _CACHED_EQUAL_LOUDNESS:
            a_weight_taps = _get_equal_loudness_filter(sample_rate)
            _CACHED_EQUAL_LOUDNESS[cache_key] = a_weight_taps
        else:
            a_weight_taps = _CACHED_EQUAL_LOUDNESS[cache_key]
    
    # Apply A-weighting for level detection
    # Use FFT convolution for efficiency
    if _HAS_TORCHAUDIO and _HAS_TORCH:
        import torch
        import torchaudio.prototype.functional as F
        
        # Convert to torch tensors
        audio_tensor = torch.tensor(padded_audio.astype(np.float32))
        taps_tensor = torch.tensor(a_weight_taps.astype(np.float32))
        
        # Apply FFT convolution
        weighted_audio = F.fftconvolve(audio_tensor, taps_tensor, mode='same').numpy()
    else:
        # Fallback to scipy
        weighted_audio = signal.fftconvolve(padded_audio, a_weight_taps, mode='same')
    
    # Calculate the absolute value for level detection
    abs_weighted = np.abs(weighted_audio)
    
    # Compute maximum levels with efficient rolling window max
    # Using scipy.ndimage for big performance boost over loop
    from scipy import ndimage
    max_levels = ndimage.maximum_filter1d(abs_weighted, size=lookahead_samples+1, mode='nearest')
    
    # Calculate the gain reduction needed with soft knee (vectorized)
    # Initialize target_reduction array first (fixing the NameError)
    target_reduction = np.ones_like(max_levels)
    
    below_threshold = max_levels < threshold_linear
    in_knee = (max_levels >= threshold_linear) & (max_levels < threshold_linear * knee_range)
    above_knee = max_levels >= threshold_linear * knee_range
    
    # Below threshold - no gain reduction
    target_reduction[below_threshold] = 1.0
    
    # In the knee region - gradual gain reduction
    knee_position = (max_levels[in_knee] - threshold_linear) / (threshold_linear * (knee_range - 1))
    gain_db_reduction = knee_db * knee_position * knee_position
    target_reduction[in_knee] = np.power(10, -gain_db_reduction / 20)
    
    # Above knee - full gain reduction
    target_reduction[above_knee] = threshold_linear / max_levels[above_knee]
    
    # Apply attack and release using first-order smoothing
    # Initialize gain reduction with first target
    gain_reduction[0] = target_reduction[0]
    
    # First-order smoothing coefficients
    attack_coef = np.exp(-1.0 / attack_samples)
    release_coef = np.exp(-1.0 / release_samples)
    
    # Apply smoothing
    for i in range(1, len(padded_audio)):
        if target_reduction[i] < gain_reduction[i-1]:
            # Attack phase - fast gain reduction
            gain_reduction[i] = attack_coef * gain_reduction[i-1] + (1 - attack_coef) * target_reduction[i]
        else:
            # Release phase - slow gain increase
            gain_reduction[i] = release_coef * gain_reduction[i-1] + (1 - release_coef) * target_reduction[i]
    
    # Apply the gain reduction (removing the lookahead padding)
    limited_audio = audio * gain_reduction[lookahead_samples:]
    
    # No true peak limiting here - this is done in FFmpeg loudnorm
    # or in a separate final processing stage
    
    return limited_audio


def _apply_fade_envelope(audio, sample_rate, fade_in_seconds=10, fade_out_seconds=60):
    """Apply fade-in and fade-out with cos² envelope"""
    fade_in_samples = int(fade_in_seconds * sample_rate)
    fade_out_samples = int(fade_out_seconds * sample_rate)
    
    # Ensure fades don't exceed audio length
    if fade_in_samples + fade_out_samples > len(audio):
        # Scale down fade durations proportionally
        total_samples = len(audio)
        ratio = total_samples / (fade_in_samples + fade_out_samples)
        fade_in_samples = int(fade_in_samples * ratio * 0.5)
        fade_out_samples = int(fade_out_samples * ratio * 0.5)
    
    # Create cos² fade-in envelope
    if fade_in_samples > 0:
        fade_in = np.linspace(0, np.pi/2, fade_in_samples)
        fade_in_env = np.sin(fade_in)**2
        audio[:fade_in_samples] *= fade_in_env
    
    # Create cos² fade-out envelope
    if fade_out_samples > 0:
        fade_out = np.linspace(0, np.pi/2, fade_out_samples)
        fade_out_env = np.cos(fade_out)**2
        audio[-fade_out_samples:] *= fade_out_env
    
    return audio


def _apply_diffusion_polish(audio, sample_rate, strength=0.5):
    """Apply audio diffusion model polish"""
    # Get diffusion model with thread-safe caching and environment control
    diffusion_model = _get_diffusion_model()
    
    if diffusion_model is None:
        logger.warning("Audio diffusion polish requested but model not available")
        return audio
    
    try:
        import torch
        
        # Convert audio to PyTorch tensor with correct shape
        # Diffusion model expects (batch, channels, time)
        if len(audio.shape) == 1:  # mono
            # Shape: (samples,) -> (1, 1, samples)
            audio_tensor = torch.tensor(audio.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        elif len(audio.shape) == 2:  # stereo
            # Shape: (samples, channels) -> (1, channels, samples)
            audio_tensor = torch.tensor(audio.astype(np.float32)).permute(1, 0).unsqueeze(0)
        
        # Move to same device as model
        device = next(diffusion_model.parameters()).device
        audio_tensor = audio_tensor.to(device)
        
        # Set noise level based on strength
        steps = int(30 * strength)
        
        # Apply diffusion
        logger.info(f"Applying audio diffusion polish (steps={steps})")
        with torch.no_grad():
            polished_tensor = diffusion_model.sample(audio_tensor, steps)
        
        # Convert back to numpy and CPU
        polished_audio = polished_tensor.cpu().squeeze().numpy()
        
        # Handle stereo if needed
        if len(polished_audio.shape) > 1 and len(audio.shape) == 1:
            # Convert back to mono if original was mono
            polished_audio = polished_audio.mean(axis=0)
        elif len(polished_audio.shape) > 1 and polished_audio.shape[0] == 2:
            # Put channels back in the right order
            polished_audio = polished_audio.T
        
        # Mix with original based on strength
        result = audio * (1 - strength) + polished_audio * strength
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(result))
        if max_val > Constants.MAX_AUDIO_VALUE:
            result = result / max_val * Constants.MAX_AUDIO_VALUE
        
        # Free GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return result
        
    except Exception as e:
        logger.error(f"Error applying diffusion polish: {e}")
        logger.warning("Returning unpolished audio")
        return audio


class NoiseGenerator(SoundProfileGenerator):
    """Enhanced generator for different colors of noise."""
    
    def __init__(self, sample_rate: int, use_perlin: bool = True, modulation_depth: float = 0.08, seed: Optional[int] = None, **kwargs):
        """
        Initialize the noise generator.
        
        Args:
            sample_rate: Audio sample rate
            use_perlin: Whether to use Perlin noise for more natural variations
            modulation_depth: Depth of modulation for dynamic variation
            seed: Random seed for reproducibility
            **kwargs: Additional parameters
        """
        super().__init__(sample_rate, use_perlin, seed, **kwargs)
        self.modulation_depth = modulation_depth
        self.random_state = RandomStateManager.get_instance(seed)
        self.seed: Optional[int] = seed  # Explicitly define for type checkers
        
        # Additional enhancement flags
        self.use_equal_loudness = kwargs.get('use_equal_loudness', True)
        self.use_limiter = kwargs.get('use_limiter', True)
        self.use_diffusion = kwargs.get('use_diffusion', False)
        self.use_organic_drift = kwargs.get('use_organic_drift', True)
        
        # Try to import dependencies at initialization
        _try_import_dependencies()
        
        # Set up thread-safe RNG
        self._rng, self._using_gpu = _get_thread_safe_rng(seed)
        
        # Helper for efficient random number generation that avoids GPU memory pinning
        self._randn_cpu = self._get_efficient_randn()
        
    def generate(self, duration_seconds: int, **kwargs) -> np.ndarray:
        """
        Generate specified noise color with enhanced quality.
        
        Args:
            duration_seconds: Duration in seconds
            **kwargs: Additional parameters including sound_type
            
        Returns:
            Noise array
        """
        # Handle edge case of zero or negative duration
        if duration_seconds <= 0:
            return np.empty(0, dtype=np.float32)
            
        # Get sound type with default to "white"
        sound_type = kwargs.get("sound_type", "white")
        
        # Handle enum if provided
        if isinstance(sound_type, NoiseColor):
            sound_type = sound_type.value
        
        try:
            # Set up thread-safe RNG
            self._rng, self._using_gpu = _get_thread_safe_rng(self.seed)
            
            # Generate the requested noise type
            if sound_type == "white":
                noise = self.generate_white_noise(duration_seconds, **kwargs)
            elif sound_type == "pink":
                # Use IIR method by default for real-time safety, fall back to FFT if needed
                noise = self.generate_pink_noise_iir(duration_seconds, **kwargs)
            elif sound_type == "brown":
                # Use IIR method by default for real-time safety, fall back to FFT if needed
                noise = self.generate_brown_noise_iir(duration_seconds, **kwargs)
            else:
                raise ValueError(f"Unknown noise type: {sound_type}")
            
            # Apply DC removal and anti-aliasing filters
            noise = self._apply_filters(noise)
            
            # Apply equal-loudness filter if enabled
            if self.use_equal_loudness:
                noise = self._apply_equal_loudness(noise)
            
            # Apply organic micro-drift if enabled
            if self.use_organic_drift:
                noise = _apply_organic_drift(noise, self.sample_rate, self.use_perlin, self.seed)
            
            # Apply soft-knee limiter if enabled
            if self.use_limiter:
                noise = _apply_soft_knee_limiter(noise, self.sample_rate)
            
            # Apply fade-in and fade-out
            noise = _apply_fade_envelope(noise, self.sample_rate)
            
            # Apply diffusion polish if enabled
            if self.use_diffusion:
                noise = _apply_diffusion_polish(noise, self.sample_rate)
            
            # Sanitize audio
            return self.sanitize_audio(noise)
            
        except Exception as e:
            logger.error(f"Error generating {sound_type} noise: {e}")
            # Generate simple fallback noise if any error occurs
            return self._generate_fallback_noise(duration_seconds)
    
    def _generate_fallback_noise(self, duration_seconds: int) -> np.ndarray:
        """Generate simple random noise as fallback in case of errors."""
        try:
            samples = int(duration_seconds * self.sample_rate)
            return self.random_state.normal(0, 0.5, samples)
        except Exception as e:
            logger.error(f"Error generating fallback noise: {e}")
            # In case of catastrophic failure, return empty array
            return np.zeros(int(duration_seconds * self.sample_rate))
    
def _get_efficient_randn(self):
    """Return a function that efficiently generates random numbers on CPU"""
    if self._using_gpu:
        import cupy as cp
        # Capture the current RNG in a local variable to avoid thread issues
        rng = self._rng
        # Create a closure that handles the GPU->CPU transfer efficiently
        def randn_cpu(n, mean=0.0, std=1.0):
            noise = rng.normal(mean, std, n, dtype=cp.float32)
            return cp.asnumpy(noise)
        return randn_cpu
    else:
        # Capture the current RNG in a local variable to avoid thread issues
        rng = self._rng
        # For CPU, just return a function that calls the RNG directly
        def randn_cpu(n, mean=0.0, std=1.0):
            return rng.normal(mean, std, n)
        return randn_cpu

    def _apply_filters(self, audio: np.ndarray) -> np.ndarray:
        """Apply DC removal and anti-aliasing filters"""
        # Get or design filters
        with _FILTER_CACHE_LOCK:
            cache_key = f"dc_alias_{self.sample_rate}"
            if cache_key not in _CACHED_FILTERS:
                (hp_b, hp_a), (lp_b, lp_a) = _design_dc_alias_filters(self.sample_rate)
                _CACHED_FILTERS[cache_key] = ((hp_b, hp_a), (lp_b, lp_a))
            else:
                (hp_b, hp_a), (lp_b, lp_a) = _CACHED_FILTERS[cache_key]
        
        # Apply highpass for DC removal
        audio = signal.lfilter(hp_b, hp_a, audio)
        
        # Apply lowpass for anti-aliasing
        audio = signal.lfilter(lp_b, lp_a, audio)
        
        return audio
    
    def _apply_equal_loudness(self, audio: np.ndarray) -> np.ndarray:
        """Apply ISO-226 equal-loudness filter"""
        # Get or compute equal loudness filter taps
        with _FILTER_CACHE_LOCK:
            cache_key = f"equal_loudness_{self.sample_rate}"
            if cache_key not in _CACHED_EQUAL_LOUDNESS:
                equal_loudness_taps = _get_equal_loudness_filter(self.sample_rate)
                _CACHED_EQUAL_LOUDNESS[cache_key] = equal_loudness_taps
            else:
                equal_loudness_taps = _CACHED_EQUAL_LOUDNESS[cache_key]
        
        # Apply filter using FFT convolution for efficiency
        if _HAS_TORCHAUDIO and _HAS_TORCH:
            import torch
            import torchaudio.prototype.functional as F
            
            # Convert to torch tensors
            audio_tensor = torch.tensor(audio.astype(np.float32))
            taps_tensor = torch.tensor(equal_loudness_taps.astype(np.float32))
            
            # Apply FFT convolution
            filtered_audio = F.fftconvolve(audio_tensor, taps_tensor, mode='same').numpy()
        else:
            # Fallback to scipy
            filtered_audio = signal.fftconvolve(audio, equal_loudness_taps, mode='same')
        
        # No normalization - preserve the loudness relationships
        return filtered_audio
    
    def generate_white_noise(self, duration_seconds: int, **kwargs) -> np.ndarray:
        """
        Generate enhanced white noise with natural texture.
        Uses high-quality RNG and optional Perlin noise.
        
        Args:
            duration_seconds: Length of the sound in seconds
            **kwargs: Additional parameters
            
        Returns:
            White noise array
        """
        samples = int(duration_seconds * self.sample_rate)

        # For very long durations, process in chunks
        if samples > PerformanceConstants.MAX_DURATION_SECONDS_BEFORE_CHUNKING * self.sample_rate:
            return self._generate_white_noise_chunked(duration_seconds, **kwargs)

        try:
            if self.use_perlin and HAS_PERLIN:
                # Use Perlin noise for a more organic texture
                noise_array = generate_perlin_noise(
                    self.sample_rate, 
                    duration_seconds, 
                    octaves=6, 
                    persistence=0.7,
                    seed=self.seed
                )

                # Apply whitening filter to ensure flat spectrum using FFT
                spectrum = np.fft.rfft(noise_array)
                magnitude = np.abs(spectrum)
                phase = np.angle(spectrum)
                flat_spectrum = np.ones_like(magnitude) * np.exp(1j * phase)
                noise_array = np.fft.irfft(flat_spectrum, n=len(noise_array))

                # Normalize
                noise_array = 0.95 * noise_array / np.max(np.abs(noise_array))
            else:
                # Use high-quality RNG
                if self._using_gpu:
                    # CuPy GPU path
                    import cupy
                    noise_array = self._rng.normal(0, 0.5, samples, dtype=cupy.float32).get()
                else:
                    # NumPy CPU path
                    noise_array = self._rng.normal(0, 0.5, samples)

            # Apply subtle dynamic modulation to prevent fatigue
            # Keep backward compatibility with old interface
            modulated_noise = generate_dynamic_modulation(
                self.sample_rate,
                duration_seconds,
                depth=self.modulation_depth,
                use_perlin=self.use_perlin,
                seed=self.seed
            )
            noise_array = noise_array * modulated_noise

            return noise_array
            
        except Exception as e:
            logger.error(f"Error generating white noise: {e}")
            # Fallback to simple white noise
            return self.random_state.normal(0, 0.5, samples)

    def _generate_white_noise_chunked(self, duration_seconds: int, stream_to_disk=False, **kwargs) -> Union[np.ndarray, str]:
        """
        Generate white noise in chunks for very long durations.
        Optionally stream directly to disk for memory-constrained environments.
        
        Args:
            duration_seconds: Length of the sound in seconds
            stream_to_disk: Whether to stream audio directly to disk rather than holding in memory
            **kwargs: Additional parameters
            
        Returns:
            Union[np.ndarray, str]: Generated noise array, or path to the file if stream_to_disk=True
        """
        # Calculate total samples
        total_samples = int(duration_seconds * self.sample_rate)
        
        # Check if we should stream to disk for memory efficiency
        if stream_to_disk and _SOUNDFILE_AVAILABLE:
            try:
                import soundfile as sf
                import tempfile
                
                # Create a temporary file with unique name
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    temp_path = tmp_file.name
                
                # Open the file for writing
                with sf.SoundFile(temp_path, 'w', 
                                 samplerate=self.sample_rate,
                                 channels=1, 
                                 format='WAV',
                                 subtype='FLOAT') as f:
                    
                    # Process in chunks
                    chunk_seconds = PerformanceConstants.FFT_CHUNK_SIZE_SECONDS
                    chunk_samples = int(chunk_seconds * self.sample_rate)
                    crossfade_samples = int(PerformanceConstants.CROSSFADE_BETWEEN_CHUNKS_SECONDS * self.sample_rate)
                    
                    # Calculate number of chunks
                    num_chunks = (total_samples + chunk_samples - 1) // chunk_samples
                    
                    logger.info(f"Generating white noise in {num_chunks} chunks, streaming to disk")
                    
                    # Initialize previous chunk for crossfading
                    prev_chunk_end = None
                    
                    # Process each chunk
                    for i in range(num_chunks):
                        # Calculate chunk boundaries
                        start_idx = i * chunk_samples
                        end_idx = min(start_idx + chunk_samples, total_samples)
                        chunk_duration = (end_idx - start_idx) / self.sample_rate
                        
                        # Log progress periodically
                        if i % max(1, num_chunks // 10) == 0 or i == num_chunks - 1:
                            logger.info(f"Generating chunk {i+1}/{num_chunks} ({int(100*(i+1)/num_chunks)}%)")
                        
                        # Generate this chunk with extra samples for crossfade if not the last chunk
                        if i < num_chunks - 1:
                            current_duration = chunk_duration + crossfade_samples / self.sample_rate
                        else:
                            current_duration = chunk_duration
                        
                        # Call the non-chunked method to generate a single chunk
                        chunk = self.generate_white_noise(current_duration, **kwargs)
                        
                        # Apply crossfade if not the first chunk
                        if i > 0 and prev_chunk_end is not None:
                            # Create crossfade weights
                            fade_in = np.linspace(0, 1, crossfade_samples)
                            fade_out = 1 - fade_in
                            
                            # Apply crossfade
                            crossfaded = prev_chunk_end * fade_out + chunk[:crossfade_samples] * fade_in
                            
                            # Write the crossfaded section
                            f.write(crossfaded)
                            
                            # Write the rest of the chunk
                            if i < num_chunks - 1:
                                f.write(chunk[crossfade_samples:-crossfade_samples])
                                prev_chunk_end = chunk[-crossfade_samples:]
                            else:
                                # For the last chunk, write everything (don't skip the end)
                                f.write(chunk[crossfade_samples:])
                        else:
                            # First chunk, no crossfade needed
                            if i < num_chunks - 1:
                                f.write(chunk[:-crossfade_samples])
                                prev_chunk_end = chunk[-crossfade_samples:]
                            else:
                                f.write(chunk)
                
                # Return the path to the file
                logger.info(f"White noise written to temporary file: {temp_path}")
                return temp_path
                
            except Exception as e:
                logger.error(f"Error streaming to disk: {e}")
                # Fall back to in-memory processing if streaming fails
                logger.warning("Falling back to in-memory processing")
        
        try:
            # Process in chunks
            chunk_seconds = PerformanceConstants.FFT_CHUNK_SIZE_SECONDS
            chunk_samples = int(chunk_seconds * self.sample_rate)
            crossfade_samples = int(PerformanceConstants.CROSSFADE_BETWEEN_CHUNKS_SECONDS * self.sample_rate)
            
            # Calculate number of chunks
            num_chunks = (total_samples + chunk_samples - 1) // chunk_samples
            
            # Create result array
            result = np.zeros(total_samples)
            
            logger.info(f"Generating white noise in {num_chunks} chunks")
            
            # Process each chunk
            for i in range(num_chunks):
                # Calculate chunk boundaries with overlap for crossfade
                start_idx = i * chunk_samples
                end_idx = min(start_idx + chunk_samples + crossfade_samples, total_samples)
                chunk_duration = (end_idx - start_idx) / self.sample_rate
                
                # Log progress periodically
                if i % max(1, num_chunks // 10) == 0 or i == num_chunks - 1:
                    logger.info(f"Generating chunk {i+1}/{num_chunks} ({int(100*(i+1)/num_chunks)}%)")
                
                # Generate this chunk with a different seed for each chunk
                chunk_seed = self.seed + i if self.seed is not None else None
                # Temporarily adjust our seed for this chunk
                original_seed = self.seed
                self.seed = chunk_seed
                
                # Call the non-chunked method
                chunk = self.generate_white_noise(chunk_duration, **kwargs)
                
                # Restore original seed
                self.seed = original_seed
                
                # Apply crossfade if not the first chunk
                if i > 0:
                    # Overlap with previous chunk
                    overlap_start = start_idx
                    overlap_end = min(start_idx + crossfade_samples, total_samples)
                    
                    # Create crossfade weights
                    fade_in = np.linspace(0, 1, overlap_end - overlap_start)
                    fade_out = 1 - fade_in
                    
                    # Apply crossfade
                    result[overlap_start:overlap_end] = (
                        result[overlap_start:overlap_end] * fade_out + 
                        chunk[:overlap_end - overlap_start] * fade_in
                    )
                    
                    # Add the non-overlapping part
                    if overlap_end < end_idx:
                        result[overlap_end:end_idx] = chunk[overlap_end - overlap_start:end_idx - overlap_start]
                else:
                    # First chunk, no crossfade needed
                    result[start_idx:end_idx] = chunk[:end_idx - start_idx]
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating chunked white noise: {e}")
            # Fall back to simple white noise - might be shorter than requested if we can't generate the full duration
            fallback_duration = min(duration_seconds, 60.0)  # Limit fallback to 60 seconds
            logger.warning(f"Falling back to {fallback_duration}s of simple white noise")
            return self.random_state.normal(0, 0.5, int(fallback_duration * self.sample_rate))

    def generate_pink_noise_iir(self, duration_seconds: int, **kwargs) -> np.ndarray:
        """
        Generate pink noise using recursive IIR filter.
        This creates an accurate 1/f spectrum and is suitable for real-time processing.
        Based on Robert Bristow-Johnson's audio EQ cookbook and the Kellet/Voss algorithm.
        
        Args:
            duration_seconds: Length of the sound in seconds
            **kwargs: Additional parameters
            
        Returns:
            Pink noise array
        """
        samples = int(duration_seconds * self.sample_rate)
        
        try:
            # Use high-quality RNG for white noise
            # Use efficient RNG with CPU output
            # For longer samples, generate in chunks to avoid memory issues
            if samples > 1_048_576:  # About 20 seconds at 48kHz
                chunk_size = 1_048_576
                white = np.zeros(samples, dtype=np.float32)
                
                for offset in range(0, samples, chunk_size):
                    end = min(offset + chunk_size, samples)
                    length = end - offset
                    white[offset:end] = self._randn_cpu(length, mean=0, std=1.0)
            else:
                white = self._randn_cpu(samples, mean=0, std=1.0)
            
            # Coefficients for Paul Kellet's 7-pole pink noise filter
            # This gives better accuracy than the simplified version
            b = [0.99886, -1.99543, 0.98639, -0.97182, 0.95400, -0.93254]
            a = [1.0, -1.99772, 0.99733, -0.99518, 0.99222, -0.98848]
            
            # Apply IIR filter efficiently using scipy.signal.lfilter
            pink_noise = signal.lfilter(b, a, white)
            
            # Normalize
            pink_noise = 0.5 * pink_noise / np.std(pink_noise)
            
            # Apply subtle dynamic modulation to prevent fatigue
            modulated_noise = generate_dynamic_modulation(
                self.sample_rate,
                duration_seconds,
                depth=self.modulation_depth,
                use_perlin=self.use_perlin,
                seed=self.seed
            )
            pink_noise = pink_noise * modulated_noise
            
            return pink_noise
            
        except Exception as e:
            logger.error(f"Error generating pink noise (IIR): {e}")
            # Fall back to FFT-based implementation
            return self.generate_pink_noise_fft(duration_seconds, **kwargs)
    
    def generate_pink_noise_fft(self, duration_seconds: int, **kwargs) -> np.ndarray:
        """
        Generate pink noise using FFT-based spectral shaping.
        This creates more accurate 1/f spectrum than filtered white noise.
        
        Args:
            duration_seconds: Length of the sound in seconds
            **kwargs: Additional parameters
            
        Returns:
            Pink noise array
        """
        samples = int(duration_seconds * self.sample_rate)
        
        # For very long durations, process in chunks
        if samples > PerformanceConstants.MAX_DURATION_SECONDS_BEFORE_CHUNKING * self.sample_rate:
            return self._generate_noise_fft_chunked(duration_seconds, noise_type="pink", **kwargs)

        try:
            # Start with white noise using high-quality RNG
            if self.use_perlin and HAS_PERLIN:
                white = generate_perlin_noise(
                    self.sample_rate, 
                    duration_seconds, 
                    octaves=6, 
                    persistence=0.7,
                    seed=self.seed
                )
            else:
                # Use efficient RNG with CPU output
                white = self._randn_cpu(samples, mean=0, std=0.5)

            # FFT to frequency domain
            X = np.fft.rfft(white)

            # Generate frequency array
            freqs = np.fft.rfftfreq(samples, 1 / self.sample_rate)

            # Create 1/f filter (pink noise has energy proportional to 1/f)
            # Add small constant to avoid division by zero
            pink_filter = 1 / np.sqrt(freqs + 1e-6)
            
            # Set DC component to avoid extreme low frequency boost
            pink_filter[0] = pink_filter[1]

            # Apply filter while preserving phase
            magnitude = np.abs(X)
            phase = np.angle(X)
            X_pink = pink_filter * magnitude * np.exp(1j * phase)

            # Back to time domain
            pink_noise = np.fft.irfft(X_pink, n=samples)

            # Normalize
            pink_noise = 0.95 * pink_noise / np.max(np.abs(pink_noise))

            # Apply subtle dynamic modulation to prevent fatigue
            modulated_noise = generate_dynamic_modulation(
                self.sample_rate,
                duration_seconds,
                depth=self.modulation_depth,
                use_perlin=self.use_perlin,
                seed=self.seed
            )
            pink_noise = pink_noise * modulated_noise

            return pink_noise
            
        except Exception as e:
            logger.error(f"Error generating pink noise: {e}")
            # Fallback to simple noise
            return self.random_state.normal(0, 0.5, samples)

    def generate_brown_noise_iir(self, duration_seconds: int, **kwargs) -> np.ndarray:
        """
        Generate brown/red noise using a recursive IIR filter.
        Brown noise has energy proportional to 1/f²
        This approach is suitable for real-time processing.
        
        Args:
            duration_seconds: Length of the sound in seconds
            **kwargs: Additional parameters
            
        Returns:
            Brown noise array
        """
        samples = int(duration_seconds * self.sample_rate)
        
        try:
            # Use high-quality RNG for white noise
            if self._using_gpu:
                # CuPy GPU path - allocate in chunks to avoid OOM
                import cupy as cp
                chunk_size = min(samples, 1_048_576)  # 1M samples per chunk (about 20 seconds at 48kHz)
                white = np.zeros(samples, dtype=np.float32)
                
                for offset in range(0, samples, chunk_size):
                    end = min(offset + chunk_size, samples)
                    length = end - offset
                    # Generate on GPU
                    noise = self._rng.standard_normal(length, dtype=cp.float32)
                    # Transfer to CPU
                    white[offset:end] = cp.asnumpy(noise)
            else:
                # NumPy CPU path
                white = self._rng.normal(0, 1.0, samples)
            
            # Simple yet effective leaky integrator (first-order IIR filter)
            # Used scipy.signal.lfilter for efficiency
            a = 0.99  # Leaky integrator coefficient
            
            # Apply IIR filter - [1-a] is the feed-forward coefficient, [1, -a] is feedback
            # This is a simple one-pole filter
            brown_noise = signal.lfilter([1-a], [1, -a], white)
            
            # Apply high-pass filter to avoid DC build-up
            # Simple first-order high-pass using scipy.signal.lfilter
            hp_a = 0.995  # Very close to 1 for gentle high-pass (~0.5 Hz at 48 kHz)
            brown_noise = signal.lfilter([1, -1], [1, -hp_a], brown_noise)
            
            # Normalize
            brown_noise = 0.5 * brown_noise / np.std(brown_noise)
            
            # Apply subtle dynamic modulation to prevent fatigue
            modulated_noise = generate_dynamic_modulation(
                self.sample_rate,
                duration_seconds,
                depth=self.modulation_depth,
                use_perlin=self.use_perlin,
                seed=self.seed
            )
            brown_noise = brown_noise * modulated_noise
            
            return brown_noise
            
        except Exception as e:
            logger.error(f"Error generating brown noise (IIR): {e}")
            # Fall back to FFT-based implementation
            return self.generate_brown_noise_fft(duration_seconds, **kwargs)
    
    def generate_brown_noise_fft(self, duration_seconds: int, **kwargs) -> np.ndarray:
        """
        Generate brown/red noise using FFT-based spectral shaping.
        Brown noise has energy proportional to 1/f²
        
        Args:
            duration_seconds: Length of the sound in seconds
            **kwargs: Additional parameters
            
        Returns:
            Brown noise array
        """
        samples = int(duration_seconds * self.sample_rate)
        
        # For very long durations, process in chunks
        if samples > PerformanceConstants.MAX_DURATION_SECONDS_BEFORE_CHUNKING * self.sample_rate:
            return self._generate_noise_fft_chunked(duration_seconds, noise_type="brown", **kwargs)

        try:
            # Start with white noise using high-quality RNG
            if self.use_perlin and HAS_PERLIN:
                white = generate_perlin_noise(
                    self.sample_rate, 
                    duration_seconds, 
                    octaves=6, 
                    persistence=0.7,
                    seed=self.seed
                )
            else:
                if self._using_gpu:
                    # CuPy GPU path
                    import cupy
                    white = self._rng.normal(0, 0.5, samples, dtype=cupy.float32).get()
                else:
                    # NumPy CPU path
                    white = self._rng.normal(0, 0.5, samples)

            # FFT to frequency domain
            X = np.fft.rfft(white)

            # Generate frequency array
            freqs = np.fft.rfftfreq(samples, 1 / self.sample_rate)

            # Create 1/f² filter (brown noise has energy proportional to 1/f²)
            # Add small constant to avoid division by zero
            brown_filter = 1 / (freqs + 1e-6)
            
            # Set DC component to avoid extreme low frequency boost
            brown_filter[0] = brown_filter[1]

            # Apply filter while preserving phase
            magnitude = np.abs(X)
            phase = np.angle(X)
            X_brown = brown_filter * magnitude * np.exp(1j * phase)

            # Back to time domain
            brown_noise = np.fft.irfft(X_brown, n=samples)

            # Normalize
            brown_noise = 0.95 * brown_noise / np.max(np.abs(brown_noise))

            # Apply subtle dynamic modulation to prevent fatigue
            modulated_noise = generate_dynamic_modulation(
                self.sample_rate,
                duration_seconds,
                depth=self.modulation_depth,
                use_perlin=self.use_perlin,
                seed=self.seed
            )
            brown_noise = brown_noise * modulated_noise

            return brown_noise
            
        except Exception as e:
            logger.error(f"Error generating brown noise: {e}")
            # Fallback to simple noise with low-pass filter
            fallback = self.random_state.normal(0, 0.5, samples)
            try:
                # Apply simple low-pass filter to approximate brown noise
                b, a = signal.butter(2, 200 / (self.sample_rate / 2), 'low')
                filtered = signal.lfilter(b, a, fallback)
                return filtered
            except:
                return fallback
        
    def _generate_noise_fft_chunked(self, duration_seconds: int, noise_type: str, **kwargs) -> np.ndarray:
        """Generate colored noise in chunks for very long durations."""
        # Calculate total samples
        total_samples = int(duration_seconds * self.sample_rate)
        
        try:
            # Process in chunks
            chunk_seconds = PerformanceConstants.FFT_CHUNK_SIZE_SECONDS
            chunk_samples = int(chunk_seconds * self.sample_rate)
            crossfade_samples = int(PerformanceConstants.CROSSFADE_BETWEEN_CHUNKS_SECONDS * self.sample_rate)
            
            # Calculate number of chunks
            num_chunks = (total_samples + chunk_samples - 1) // chunk_samples
            
            # Create result array
            result = np.zeros(total_samples)
            
            logger.info(f"Generating {noise_type} noise in {num_chunks} chunks of {chunk_seconds}s each")
            
            # Process each chunk
            for i in range(num_chunks):
                # Calculate chunk boundaries with overlap for crossfade
                start_idx = i * chunk_samples
                end_idx = min(start_idx + chunk_samples + crossfade_samples, total_samples)
                chunk_duration = (end_idx - start_idx) / self.sample_rate
                
                # Log progress periodically
                if i % max(1, num_chunks // 10) == 0 or i == num_chunks - 1:
                    logger.info(f"Generating chunk {i+1}/{num_chunks} ({int(100*(i+1)/num_chunks)}%)")
                
                # Generate this chunk based on noise type - with a different seed for each chunk
                chunk_seed = self.seed + i if self.seed is not None else None
                # Temporarily adjust our seed for this chunk
                original_seed = self.seed
                self.seed = chunk_seed
                
                # Call the non-chunked functions directly to avoid recursion stack overflow
                chunk_samples = int(chunk_duration * self.sample_rate)
                
                # Start with white noise
                if self.use_perlin and HAS_PERLIN:
                    white = generate_perlin_noise(
                        self.sample_rate, 
                        chunk_duration, 
                        octaves=6, 
                        persistence=0.7,
                        seed=self.seed
                    )
                else:
                    # Use efficient RNG with CPU output
                    white = self._randn_cpu(chunk_samples, mean=0, std=0.5)
                
                # Process based on noise type
                if noise_type == "pink":
                    # FFT to frequency domain
                    X = np.fft.rfft(white)
                    # Generate frequency array
                    freqs = np.fft.rfftfreq(chunk_samples, 1 / self.sample_rate)
                    # Create 1/f filter (pink noise)
                    pink_filter = 1 / np.sqrt(freqs + 1e-6)
                    pink_filter[0] = pink_filter[1]  # Set DC component
                    # Apply filter while preserving phase
                    magnitude = np.abs(X)
                    phase = np.angle(X)
                    X_pink = pink_filter * magnitude * np.exp(1j * phase)
                    # Back to time domain
                    chunk = np.fft.irfft(X_pink, n=chunk_samples)
                    chunk = 0.95 * chunk / np.max(np.abs(chunk))
                    
                elif noise_type == "brown":
                    # FFT to frequency domain
                    X = np.fft.rfft(white)
                    # Generate frequency array
                    freqs = np.fft.rfftfreq(chunk_samples, 1 / self.sample_rate)
                    # Create 1/f² filter (brown noise)
                    brown_filter = 1 / (freqs + 1e-6)
                    brown_filter[0] = brown_filter[1]  # Set DC component
                    # Apply filter while preserving phase
                    magnitude = np.abs(X)
                    phase = np.angle(X)
                    X_brown = brown_filter * magnitude * np.exp(1j * phase)
                    # Back to time domain
                    chunk = np.fft.irfft(X_brown, n=chunk_samples)
                    chunk = 0.95 * chunk / np.max(np.abs(chunk))
                    
                else:  # white noise
                    # For white noise, just use the generated noise
                    chunk = white
                
                # Apply subtle dynamic modulation to prevent fatigue
                modulated_noise = generate_dynamic_modulation(
                    self.sample_rate,
                    chunk_duration,
                    depth=self.modulation_depth,
                    use_perlin=self.use_perlin,
                    seed=self.seed
                )
                chunk = chunk * modulated_noise
                
                # Restore original seed
                self.seed = original_seed
                
                # Apply crossfade if not the first chunk
                if i > 0:
                    # Overlap with previous chunk
                    overlap_start = start_idx
                    overlap_end = min(start_idx + crossfade_samples, total_samples)
                    
                    # Only proceed if overlap region exists
                    if overlap_end > overlap_start:
                        # Create crossfade weights
                        fade_in = np.linspace(0, 1, overlap_end - overlap_start)
                        fade_out = 1 - fade_in
                        
                        # Apply crossfade
                        result[overlap_start:overlap_end] = (
                            result[overlap_start:overlap_end] * fade_out + 
                            chunk[:overlap_end - overlap_start] * fade_in
                        )
                        
                        # Add the non-overlapping part
                        if overlap_end < end_idx:
                            result[overlap_end:end_idx] = chunk[overlap_end - overlap_start:end_idx - overlap_start]
                else:
                    # First chunk, no crossfade needed
                    result[start_idx:end_idx] = chunk[:end_idx - start_idx]
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating chunked {noise_type} noise: {e}")
            # Fall back to simple noise
            fallback_duration = min(duration_seconds, 60.0)  # Limit fallback to 60 seconds
            logger.warning(f"Falling back to {fallback_duration}s of simple noise")
            return self.random_state.normal(0, 0.5, int(fallback_duration * self.sample_rate))

    def export_master(self, audio: np.ndarray, path: str, format: str = 'wav', 
                    target_lufs: float = -23, lra: float = 7, true_peak: float = -3) -> str:
        """
        Export the audio with proper loudness normalization using FFmpeg.
        
        Args:
            audio: Audio array to export
            path: Output file path
            format: Output format ('wav' or 'mp3')
            target_lufs: Target integrated loudness in LUFS
            lra: Target loudness range in LU
            true_peak: Target maximum true peak in dBTP
            
        Returns:
            Path to the exported file
        """
        # Check if soundfile is available for saving intermediary file
        global _SOUNDFILE_AVAILABLE
        if not _SOUNDFILE_AVAILABLE:
            _try_import_dependencies()
        
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False) as tmp_file:
                temp_path = tmp_file.name
            
            # Save audio to temporary file with 24-bit or 32-bit float precision
            if _SOUNDFILE_AVAILABLE:
                import soundfile as sf
                # Use 32-bit float for maximum precision
                sf.write(temp_path, audio, self.sample_rate, subtype='FLOAT')
            else:
                # Fallback to scipy.io.wavfile
                from scipy.io import wavfile
                # Convert to 32-bit float for maximum precision
                wavfile.write(temp_path, self.sample_rate, audio.astype(np.float32))
            
            # Construct FFmpeg command
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",  # Overwrite output file if it exists
                "-i", temp_path,  # Input file
                "-af", f"loudnorm=I={target_lufs}:LRA={lra}:TP={true_peak}:print_format=summary",  # Loudness normalization
                "-ar", str(self.sample_rate),  # Output sample rate
                "-c:a", "pcm_s16le" if format == 'wav' else "libmp3lame",  # Codec
                "-b:a", "320k" if format == 'mp3' else None,  # Bitrate for MP3
                path  # Output file
            ]
            
            # Remove None values
            ffmpeg_cmd = [arg for arg in ffmpeg_cmd if arg is not None]
            
            # Run FFmpeg
            logger.info(f"Running FFmpeg for loudness normalization: {' '.join(ffmpeg_cmd)}")
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
            
            # Check for errors
            if result.returncode != 0:
                logger.error(f"FFmpeg error: {result.stderr}")
                # Fallback to the temporary file if FFmpeg fails
                import shutil
                shutil.copy(temp_path, path)
                logger.warning(f"Copied unnormalized audio to {path}")
            else:
                logger.info(f"Successfully exported normalized audio to {path}")
                # Log the loudness information from FFmpeg
                loudness_info = [line for line in result.stderr.split('\n') if 'loudnorm' in line]
                for line in loudness_info:
                    logger.info(line)
            
            # Clean up temp file
            try:
                os.remove(temp_path)
            except OSError:
                pass
            
            return path
            
        except Exception as e:
            logger.error(f"Error exporting audio: {e}")
            # Try direct export without normalization as fallback
            try:
                if _SOUNDFILE_AVAILABLE:
                    import soundfile as sf
                    sf.write(path, audio, self.sample_rate)
                else:
                    # Fallback to scipy.io.wavfile
                    from scipy.io import wavfile
                    int_audio = (audio * 32767).astype(np.int16)
                    wavfile.write(path, self.sample_rate, int_audio)
                logger.warning(f"Exported unnormalized audio to {path}")
                return path
            except Exception as e2:
                logger.error(f"Error in fallback export: {e2}")
                return ""