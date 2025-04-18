"""
Factory for creating sound generators.
"""

from typing import Dict, Type, Optional, Any

from sound_profiles.base import SoundProfileGenerator
from sound_profiles.noise import NoiseGenerator
from sound_profiles.natural import NaturalSoundGenerator
from sound_profiles.womb import WombSoundGenerator

class SoundGeneratorFactory:
    """
    Factory for creating sound generators.
    """
    
    _generators: Dict[str, Type[SoundProfileGenerator]] = {
        "noise": NoiseGenerator,
        "natural": NaturalSoundGenerator,
        "womb": WombSoundGenerator,
    }
    
    # Mapping from sound types to generator types
    _sound_to_generator_map: Dict[str, str] = {
        # Noise generator sounds
        "white": "noise",
        "pink": "noise",
        "brown": "noise",
        
        # Natural sounds
        "heartbeat": "natural",
        "shushing": "natural",
        "fan": "natural",
        
        # Womb sounds
        "womb": "womb",
        "umbilical_swish": "womb",
    }
    
    @classmethod
    def register_generator(cls, name: str, generator_class: Type[SoundProfileGenerator]):
        """
        Register a new generator class.
        
        Args:
            name: Name identifier for the generator
            generator_class: Generator class to register
        """
        cls._generators[name] = generator_class
    
    @classmethod
    def register_sound_type(cls, sound_type: str, generator_type: str):
        """
        Register a mapping from sound type to generator type.
        
        Args:
            sound_type: The sound type identifier
            generator_type: The generator type to use for this sound
        """
        cls._sound_to_generator_map[sound_type] = generator_type
    
    @classmethod
    def get_generator_type_for_sound(cls, sound_type: str) -> str:
        """
        Get the appropriate generator type for a given sound type.
        
        Args:
            sound_type: The sound type to look up
            
        Returns:
            The generator type to use
            
        Raises:
            ValueError: If sound_type is not registered
        """
        if sound_type not in cls._sound_to_generator_map:
            raise ValueError(f"Unknown sound type: {sound_type}. "
                            f"Available sound types: {list(cls._sound_to_generator_map.keys())}")
        
        return cls._sound_to_generator_map[sound_type]
    
    @classmethod
    def create_generator(cls, 
                        generator_type: str, 
                        sample_rate: int, 
                        use_perlin: bool = True, 
                        seed: Optional[int] = None, 
                        **kwargs) -> SoundProfileGenerator:
        """
        Create a generator instance.
        
        Args:
            generator_type: Type of generator to create
            sample_rate: Audio sample rate
            use_perlin: Whether to use Perlin noise
            seed: Random seed
            **kwargs: Additional arguments for the generator
            
        Returns:
            Instance of the requested generator
            
        Raises:
            ValueError: If the generator type is not registered
        """
        if generator_type not in cls._generators:
            raise ValueError(f"Unknown generator type: {generator_type}. "
                            f"Available generators: {list(cls._generators.keys())}")
        
        generator_class = cls._generators[generator_type]
        return generator_class(sample_rate, use_perlin, seed=seed, **kwargs)
    
    @classmethod
    def create_generator_for_sound(cls,
                                  sound_type: str,
                                  sample_rate: int,
                                  use_perlin: bool = True,
                                  seed: Optional[int] = None,
                                  **kwargs) -> SoundProfileGenerator:
        """
        Create the appropriate generator for a given sound type.
        
        Args:
            sound_type: Type of sound to generate
            sample_rate: Audio sample rate
            use_perlin: Whether to use Perlin noise
            seed: Random seed
            **kwargs: Additional arguments for the generator
            
        Returns:
            Instance of the appropriate generator for the sound type
        """
        generator_type = cls.get_generator_type_for_sound(sound_type)
        return cls.create_generator(generator_type, sample_rate, use_perlin, seed, **kwargs)