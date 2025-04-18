"""
Centralized random state management for reproducibility.
"""

import random
import numpy as np
import logging

logger = logging.getLogger("BabySleepSoundGenerator")

class RandomStateManager:
    """Centralized manager for random state to ensure reproducibility."""
    
    _instance = None
    
    @classmethod
    def get_instance(cls, seed=None):
        """Singleton access method."""
        if cls._instance is None:
            cls._instance = RandomStateManager(seed)
        elif seed is not None and seed != cls._instance.seed:
            # If a new seed is provided, update the existing instance
            cls._instance.set_seed(seed)
            logger.info(f"Random seed updated to: {seed}")
        return cls._instance
    
    def __init__(self, seed=None):
        """Initialize with an optional seed."""
        self.seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        logger.info(f"Initializing random state with seed: {self.seed}")
        self._random = random.Random(self.seed)
        self._numpy_random = np.random.RandomState(self.seed)
        
    def set_seed(self, seed):
        """Change the seed."""
        self.seed = seed
        self._random.seed(seed)
        self._numpy_random.seed(seed)
        
    def randint(self, a, b):
        """Get a random integer in range [a, b]."""
        return self._random.randint(a, b)
        
    def random(self):
        """Get a random float in range [0.0, 1.0)."""
        return self._random.random()
    
    def choice(self, seq):
        """Choose a random element from a non-empty sequence."""
        return self._random.choice(seq)
    
    def sample(self, population, k):
        """Return a k length list of unique elements chosen from population sequence."""
        return self._random.sample(population, k)
    
    def shuffle(self, x):
        """Shuffle list x in place."""
        return self._random.shuffle(x)
    
    def normal(self, loc=0.0, scale=1.0, size=None):
        """Get normally distributed random numbers."""
        return self._numpy_random.normal(loc, scale, size)
    
    def randn(self, *args):
        """Get standard normally distributed random numbers."""
        return self._numpy_random.randn(*args)
    
    def uniform(self, low=0.0, high=1.0, size=None):
        """Get uniformly distributed random numbers."""
        return self._numpy_random.uniform(low, high, size)