"""
Base sound profile generation classes and abstract interfaces.
"""

from abc import ABC, abstractmethod
import logging
import numpy as np
from typing import Optional, Dict, Any, Union


logger = logging.getLogger("BabySleepSoundGenerator")


class SoundProfileGenerator(ABC):
    """
    Abstract base class for all sound profile generators.
    Defines the common interface for generating sound profiles.
    """
    
    def __init_subclass__(cls, **kwargs):
        """Validate that subclasses implement all required methods."""
        super().__init_subclass__(**kwargs)
        
        # Check that the generate method is implemented
        if 'generate' not in cls.__dict__:
            raise TypeError(f"Class {cls.__name__} must implement abstract method 'generate'")
    
    def __init__(self, sample_rate: int, use_perlin: bool = True, seed: Optional[int] = None, **kwargs):
        """
        Initialize the sound profile generator.
        
        Args:
            sample_rate: Audio sample rate
            use_perlin: Whether to use Perlin noise for more natural variations
            seed: Random seed for reproducibility
            **kwargs: Additional parameters
        """
        self.sample_rate = sample_rate
        self.use_perlin = use_perlin
        self.seed = seed
        
        # Process any additional parameters
        for key, value in kwargs.items():
            setattr(self, key, value)
        
    @abstractmethod
    def generate(self, duration_seconds: int, **kwargs) -> np.ndarray:
        """
        Generate a sound profile.
        
        Args:
            duration_seconds: Duration in seconds
            **kwargs: Additional generation parameters
            
        Returns:
            Sound profile as numpy array
        """
        pass
        
    def validate_parameters(self, **kwargs) -> Dict[str, Any]:
        """
        Validate parameters for generation and apply defaults where needed.
        
        Args:
            **kwargs: Parameters to validate
            
        Returns:
            Dictionary with validated parameters
        """
        # Base implementation just returns the parameters unchanged
        # Subclasses should override to add specific validation logic
        return kwargs
        
    def sanitize_audio(self, audio: np.ndarray, normalize: bool = True, max_amplitude: float = 0.95) -> np.ndarray:
        """
        Sanitize audio data to ensure it's properly formatted and in a valid range.
        
        Args:
            audio: Audio data to sanitize
            normalize: Whether to normalize the audio to the specified maximum amplitude
            max_amplitude: Maximum amplitude for normalization
            
        Returns:
            Sanitized audio data
        """
        try:
            # Make a copy to avoid modifying the original
            audio_copy = audio.copy()
            
            # First, check for NaN or infinite values and replace them
            if np.isnan(audio_copy).any():
                logger.warning("NaN values found in audio data, replacing with zeros")
                audio_copy = np.nan_to_num(audio_copy, nan=0.0)
                
            if np.isinf(audio_copy).any():
                logger.warning("Infinite values found in audio data, capping to valid range")
                audio_copy = np.nan_to_num(audio_copy, posinf=max_amplitude, neginf=-max_amplitude)
            
            # Check for very large values that could cause overflow
            max_val = np.max(np.abs(audio_copy))
            if max_val > 1e6:
                logger.warning(f"Very large values detected in audio ({max_val}), scaling down to prevent overflow")
                audio_copy = audio_copy / max_val * max_amplitude
                
            # Clip to ensure safe range before conversion
            audio_copy = np.clip(audio_copy, -1.0, 1.0)
                
            # Convert to float32 safely (avoid overflow)
            if audio_copy.dtype != np.float32:
                try:
                    # Use astype with 'copy=False' to avoid doubling memory usage
                    audio_copy = audio_copy.astype(np.float32, copy=False)
                except Exception as e:
                    logger.error(f"Error converting to float32: {e}")
                    # Perform explicit check for potential overflow
                    if max_val > np.finfo(np.float32).max:
                        audio_copy = audio_copy / max_val * 0.9  # Scale down if too large
                    audio_copy = audio_copy.astype(np.float32)
                    
            # Normalize if requested
            if normalize:
                max_val = np.max(np.abs(audio_copy))
                if max_val > 0 and max_val != 1.0:  # Only normalize if needed
                    audio_copy = audio_copy / max_val * max_amplitude
                    
            # Check for any remaining issues
            if np.isnan(audio_copy).any() or np.isinf(audio_copy).any():
                logger.warning("NaN/Inf values still present after sanitization! Using zeros as fallback.")
                audio_copy = np.zeros_like(audio_copy, dtype=np.float32)
                    
            return audio_copy
            
        except Exception as e:
            logger.error(f"Error sanitizing audio: {e}")
            # Return zeros array if sanitization fails (safer than returning potentially corrupted audio)
            shape = audio.shape if hasattr(audio, 'shape') else (len(audio),)
            return np.zeros(shape, dtype=np.float32)
            
    def get_effective_duration(self, requested_duration: float) -> float:
        """
        Get the effective duration for generation, accounting for any constraints.
        
        Args:
            requested_duration: Requested duration in seconds
            
        Returns:
            Effective duration to use (may be adjusted from requested)
        """
        # Base implementation returns the requested duration unchanged
        # Subclasses can override to implement max duration limitations
        return requested_duration