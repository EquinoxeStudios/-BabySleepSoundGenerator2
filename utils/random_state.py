"""
Centralized random state management for reproducibility.
Thread-safe implementation with proper initialization and cleanup.
"""

import random
import numpy as np
import logging
import threading
import os

logger = logging.getLogger("BabySleepSoundGenerator")

class RandomStateManager:
    """
    Centralized manager for random state to ensure reproducibility.
    Thread-safe implementation with proper initialization and cleanup.
    """
    
    _instance = None
    _lock = threading.Lock()
    _thread_local = threading.local()
    
    @classmethod
    def get_instance(cls, seed=None):
        """
        Singleton access method with double-checked locking pattern.
        
        Args:
            seed: Optional random seed to use
            
        Returns:
            Singleton instance of RandomStateManager
        """
        # Fast path - check without lock first
        if cls._instance is None:
            # Slow path - acquire lock and check again
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls._create_instance(seed)
                    
        # Handle seed update if needed
        if seed is not None and seed != cls._instance.seed:
            with cls._lock:
                if seed != cls._instance.seed:
                    cls._instance.set_seed(seed)
                    logger.info(f"Random seed updated to: {seed}")
                    
        return cls._instance
    
    @classmethod
    def _create_instance(cls, seed=None):
        """
        Create a new instance with proper error handling.
        
        Args:
            seed: Optional random seed to use
            
        Returns:
            New instance of RandomStateManager
        """
        try:
            return RandomStateManager(seed)
        except Exception as e:
            logger.error(f"Failed to create RandomStateManager: {e}")
            # Create with default seed as fallback
            return RandomStateManager(None)
    
    @classmethod
    def get_thread_local_rng(cls, seed=None):
        """
        Get a thread-local random number generator.
        
        Args:
            seed: Optional random seed to use
            
        Returns:
            Thread-local random number generator
        """
        if not hasattr(cls._thread_local, 'rng'):
            # Initialize thread-local RNG
            if seed is None:
                # Derive from global seed if not specified
                global_instance = cls.get_instance()
                with cls._lock:
                    seed = global_instance._random.randint(0, 2**32 - 1)
                    
            cls._thread_local.rng = cls._create_thread_local_rng(seed)
            cls._thread_local.seed = seed
            
        return cls._thread_local.rng
    
    @classmethod
    def _create_thread_local_rng(cls, seed):
        """
        Create a thread-local RNG with proper error handling.
        
        Args:
            seed: Random seed to use
            
        Returns:
            Thread-local random number generator
        """
        try:
            # Try to use Philox for higher quality random numbers
            from numpy.random import Generator, Philox
            return Generator(Philox(seed))
        except (ImportError, AttributeError):
            # Fall back to standard RandomState
            return np.random.RandomState(seed)
    
    @classmethod
    def cleanup_thread_local(cls):
        """
        Clean up thread-local resources.
        Call this when a thread is about to exit.
        """
        if hasattr(cls._thread_local, 'rng'):
            delattr(cls._thread_local, 'rng')
            delattr(cls._thread_local, 'seed')
    
    def __init__(self, seed=None):
        """
        Initialize with an optional seed.
        
        Args:
            seed: Optional random seed to use
        """
        try:
            self.seed = seed if seed is not None else self._generate_secure_seed()
            logger.info(f"Initializing random state with seed: {self.seed}")
            self._random = random.Random(self.seed)
            
            # Try to use Philox for higher quality random numbers
            try:
                from numpy.random import Generator, Philox
                self._numpy_random = Generator(Philox(self.seed))
                logger.info("Using high-quality Philox RNG")
            except (ImportError, AttributeError):
                # Fall back to standard RandomState if Philox not available
                logger.info("Philox RNG not available, using standard NumPy RandomState")
                self._numpy_random = np.random.RandomState(self.seed)
        except Exception as e:
            logger.error(f"Error initializing RandomStateManager: {e}")
            # Set fallback values
            self.seed = 0
            self._random = random.Random(0)
            self._numpy_random = np.random.RandomState(0)
    
    def _generate_secure_seed(self):
        """
        Generate a cryptographically secure random seed.
        
        Returns:
            Secure random seed
        """
        try:
            # Try to use os.urandom for better randomness
            import os
            random_bytes = os.urandom(4)
            return int.from_bytes(random_bytes, byteorder='little')
        except (OSError, NotImplementedError):
            # Fall back to less secure but still reasonable random
            return random.randint(0, 2**32 - 1)
    
    def set_seed(self, seed):
        """
        Change the seed with thread safety.
        
        Args:
            seed: New random seed to use
        """
        with self._lock:
            self.seed = seed
            self._random.seed(seed)
            
            # Reset NumPy RNG with the new seed
            try:
                from numpy.random import Generator, Philox
                self._numpy_random = Generator(Philox(self.seed))
            except (ImportError, AttributeError):
                self._numpy_random = np.random.RandomState(self.seed)
    
    def randint(self, a, b):
        """
        Get a random integer in range [a, b].
        
        Args:
            a: Lower bound (inclusive)
            b: Upper bound (inclusive)
            
        Returns:
            Random integer
        """
        with self._lock:
            return self._random.randint(a, b)
        
    def random(self):
        """
        Get a random float in range [0.0, 1.0).
        
        Returns:
            Random float
        """
        with self._lock:
            return self._random.random()
    
    def choice(self, seq):
        """
        Choose a random element from a non-empty sequence.
        
        Args:
            seq: Sequence to choose from
            
        Returns:
            Random element from sequence
        """
        with self._lock:
            return self._random.choice(seq)
    
    def sample(self, population, k):
        """
        Return a k length list of unique elements chosen from population sequence.
        
        Args:
            population: Sequence to sample from
            k: Number of elements to sample
            
        Returns:
            List of random elements
        """
        with self._lock:
            return self._random.sample(population, k)
    
    def shuffle(self, x):
        """
        Shuffle list x in place.
        
        Args:
            x: List to shuffle
        """
        with self._lock:
            return self._random.shuffle(x)
    
    def normal(self, loc=0.0, scale=1.0, size=None):
        """
        Get normally distributed random numbers.
        
        Args:
            loc: Mean of the distribution
            scale: Standard deviation
            size: Output shape
            
        Returns:
            Array of random numbers
        """
        with self._lock:
            return self._numpy_random.normal(loc, scale, size)
    
    def randn(self, *args):
        """
        Get standard normally distributed random numbers.
        
        Args:
            *args: Shape of output array
            
        Returns:
            Array of random numbers
        """
        with self._lock:
            return self._numpy_random.standard_normal(size=args)
    
    def uniform(self, low=0.0, high=1.0, size=None):
        """
        Get uniformly distributed random numbers.
        
        Args:
            low: Lower boundary
            high: Upper boundary
            size: Output shape
            
        Returns:
            Array of random numbers
        """
        with self._lock:
            return self._numpy_random.uniform(low, high, size)