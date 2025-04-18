"""
Dynamic modulation and variation for audio signals.
"""

import numpy as np
from models.constants import Constants
# Direct import from perlin_utils instead of through utils.__init__
from utils.perlin_utils import generate_perlin_noise, apply_modulation, generate_dynamic_modulation


class ModulationProcessor:
    """Processor for creating organic modulation and temporal variations."""
    
    def __init__(self, sample_rate: int, use_perlin: bool = True, depth: float = 0.08, rate: float = 0.002):
        """
        Initialize the modulation processor.
        
        Args:
            sample_rate: Audio sample rate
            use_perlin: Whether to use Perlin noise for more natural variations
            depth: Modulation depth (percentage of amplitude variation)
            rate: Modulation rate in Hz (cycles per second)
        """
        self.sample_rate = sample_rate
        self.use_perlin = use_perlin
        self.modulation_depth = depth
        self.modulation_rate = rate
                
    def apply_dynamic_modulation(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply subtle, organic modulation to prevent listener fatigue.
        This uses Perlin noise to create ultra-slow variations over time.
        
        Args:
            audio: Input audio array
            
        Returns:
            Modulated audio
        """
        samples = len(audio)
        duration_seconds = samples / self.sample_rate

        # Create very slow modulation curve
        mod_curve = generate_dynamic_modulation(
            self.sample_rate,
            duration_seconds,
            depth=self.modulation_depth,
            rate=self.modulation_rate,
            use_perlin=self.use_perlin
        )

        # Apply the modulation efficiently
        modulated_audio = apply_modulation(audio, mod_curve)

        # Normalize if needed
        max_val = np.max(np.abs(modulated_audio))
        if max_val > Constants.MAX_AUDIO_VALUE:
            modulated_audio = modulated_audio / max_val * Constants.MAX_AUDIO_VALUE

        return modulated_audio