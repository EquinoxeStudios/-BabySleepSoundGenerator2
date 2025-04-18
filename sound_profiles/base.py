"""
Base sound profile generation classes and abstract interfaces.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional


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