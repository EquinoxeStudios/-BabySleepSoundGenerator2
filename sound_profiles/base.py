"""
Base sound profile generation classes and abstract interfaces.
"""

from abc import ABC, abstractmethod
import numpy as np


class SoundProfileGenerator(ABC):
    """
    Abstract base class for all sound profile generators.
    Defines the common interface for generating sound profiles.
    """
    
    def __init__(self, sample_rate: int, use_perlin: bool = True, **kwargs):
        """
        Initialize the sound profile generator.
        
        Args:
            sample_rate: Audio sample rate
            use_perlin: Whether to use Perlin noise for more natural variations
            **kwargs: Additional parameters
        """
        self.sample_rate = sample_rate
        self.use_perlin = use_perlin
        
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