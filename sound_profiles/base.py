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
            # Convert to float32 if not already
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
                
            # Check for NaN values and replace with zeros
            if np.isnan(audio).any():
                logger.warning("NaN values found in audio data, replacing with zeros")
                audio = np.nan_to_num(audio)
                
            # Check for infinite values and replace with max values
            if np.isinf(audio).any():
                logger.warning("Infinite values found in audio data, capping to valid range")
                audio = np.clip(audio, -max_amplitude, max_amplitude)
                
            # Normalize if requested
            if normalize:
                max_val = np.max(np.abs(audio))
                if max_val > 0 and max_val != 1.0:  # Only normalize if needed
                    audio = audio / max_val * max_amplitude
                    
            return audio
            
        except Exception as e:
            logger.error(f"Error sanitizing audio: {e}")
            # Return original audio if sanitization fails
            return audio
            
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