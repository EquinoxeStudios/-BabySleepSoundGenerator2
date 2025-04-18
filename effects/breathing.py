"""
Breathing modulation effects.
"""

import numpy as np
from models.parameters import BreathingModulation
from utils.optional_imports import HAS_PERLIN
from utils.perlin_utils import generate_perlin_noise, apply_modulation


class BreathingModulator:
    """Applies breathing rhythm modulation to audio."""
    
    def __init__(self, sample_rate: int, use_perlin: bool = True):
        """
        Initialize the breathing modulator.
        
        Args:
            sample_rate: Audio sample rate
            use_perlin: Whether to use Perlin noise for more natural variations
        """
        self.sample_rate = sample_rate
        self.use_perlin = use_perlin and HAS_PERLIN
        
    def apply_breathing_modulation(
        self, audio: np.ndarray, params: BreathingModulation
    ) -> np.ndarray:
        """
        Apply subtle modulation matching maternal breathing rhythm.

        Args:
            audio: Input audio array
            params: Parameters for breathing modulation

        Returns:
            Modulated audio
        """
        if not params or not params.enabled:
            return audio

        # Extract parameters
        cpm = params.cycles_per_minute

        # Convert to Hz
        breathing_freq = cpm / 60.0

        # Generate the modulation envelope
        samples = len(audio)
        duration_seconds = samples / self.sample_rate

        # Time points
        t = np.arange(samples) / self.sample_rate

        if HAS_PERLIN and self.use_perlin:
            # Generate perlin noise sampled at breathing frequency
            # for more natural, organic variations
            perlin = generate_perlin_noise(
                self.sample_rate,
                duration_seconds / 5, 
                octaves=1, 
                persistence=0.5
            )

            # Create breathing cycles with slight organic variation
            n_cycles = int(samples / self.sample_rate * breathing_freq) + 1
            indices = np.linspace(0, len(perlin) - 1, n_cycles * 10)
            indices = np.clip(indices.astype(int), 0, len(perlin) - 1)
            cycle_variation = perlin[indices] * 0.1  # 10% variation

            # Create breathing pattern with base frequency + variation
            breathing = np.sin(2 * np.pi * breathing_freq * t)

            # Apply variations by warping the time dimension slightly
            warped_breathing = np.zeros_like(breathing)
            warp_amount = 0.1  # 10% time warping at most

            # Create warped time vector
            warped_t = t.copy()
            warp_freq = breathing_freq / 5  # Slower variation
            warp = warp_amount * np.sin(2 * np.pi * warp_freq * t)
            warped_t += warp

            # Sample breathing pattern at warped time points
            breathing_cycle = np.sin(2 * np.pi * breathing_freq * warped_t)

            # Scale to subtle modulation range (5% intensity variation)
            modulation = 1.0 + 0.05 * breathing_cycle
        else:
            # Simple sinusoidal modulation if Perlin not available
            modulation = 1.0 + 0.05 * np.sin(2 * np.pi * breathing_freq * t)

        # Apply modulation efficiently
        output = apply_modulation(audio, modulation)

        return output