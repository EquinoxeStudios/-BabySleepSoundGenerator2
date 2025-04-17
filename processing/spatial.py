"""
Spatial audio processing and HRTF-based spatialization.
"""

import logging
import numpy as np
from scipy import signal
from functools import lru_cache

from utils.optional_imports import HAS_LIBROSA

logger = logging.getLogger("BabySleepSoundGenerator")


class SpatialProcessor:
    """Processor for spatial audio and HRTF-based spatialization."""
    
    def __init__(self, sample_rate: int, use_hrtf: bool = True):
        """
        Initialize the spatial processor.
        
        Args:
            sample_rate: Audio sample rate
            use_hrtf: Whether to use HRTF-based spatialization
        """
        self.sample_rate = sample_rate
        self.use_hrtf = use_hrtf and HAS_LIBROSA
        
        # Load HRTF data if available and requested
        self.hrtf_data = None
        if self.use_hrtf:
            try:
                # Try to load MIT KEMAR HRTF dataset if available
                # This is a simplified version - ideally we'd have a proper HRTF database
                self.hrtf_data = self._load_hrtf_data()
            except Exception as e:
                logger.warning(f"Could not load HRTF data: {e}")
                self.use_hrtf = False
        
    @lru_cache(maxsize=8)
    def _load_hrtf_data(self) -> dict:
        """
        Load HRTF data for spatial audio processing.
        In a production environment, you would use a proper HRTF database.
        This is a simplified placeholder implementation.
        
        Returns:
            Dictionary of HRTF impulse responses indexed by angle
        """
        if not HAS_LIBROSA:
            return None

        logger.info("Loading HRTF data...")
        # Create a dict to hold impulse responses for different angles
        hrtf_data = {}

        # Generate synthetic HRTF filters for different angles
        angles = np.linspace(0, 350, 36)  # 36 angles around the head

        for angle in angles:
            # Convert angle to radians
            angle_rad = np.deg2rad(angle)

            # Create impulse length (typically ~512 samples for HRTF)
            impulse_length = 512

            # Create synthetic HRTF for left and right ears
            # This is a simplified model - real HRTFs are measured
            left_delay = 0.2 * np.sin(angle_rad)  # Delay in ms
            right_delay = -0.2 * np.sin(angle_rad)  # Delay in ms

            # Convert delay to samples
            left_delay_samples = int(left_delay * self.sample_rate / 1000)
            right_delay_samples = int(right_delay * self.sample_rate / 1000)

            # Create impulse responses
            left_ir = np.zeros(impulse_length)
            right_ir = np.zeros(impulse_length)

            # Set impulse position with delay
            center_idx = impulse_length // 2
            left_idx = max(0, min(impulse_length - 1, center_idx + left_delay_samples))
            right_idx = max(0, min(impulse_length - 1, center_idx + right_delay_samples))

            left_ir[left_idx] = 1.0
            right_ir[right_idx] = 1.0

            # Apply smoothing
            def gaussian_window(M, std):
                n = np.arange(0, M) - (M - 1.0) / 2.0
                return np.exp(-0.5 * (n / std) ** 2)
            
            left_ir = gaussian_window(impulse_length, std=5) * left_ir
            right_ir = gaussian_window(impulse_length, std=5) * right_ir

            # Apply frequency-dependent attenuation based on angle
            # (Simplified model of head shadowing)
            b, a = signal.butter(4, 2000 / (self.sample_rate / 2), "low")

            if angle > 90 and angle < 270:
                # Sound source is on the left, attenuate right ear high frequencies
                right_ir = signal.lfilter(b, a, right_ir)
            elif angle < 90 or angle > 270:
                # Sound source is on the right, attenuate left ear high frequencies
                left_ir = signal.lfilter(b, a, left_ir)

            # Store the HRTF pair
            hrtf_data[angle] = (left_ir, right_ir)

        logger.info(f"HRTF data loaded for {len(hrtf_data)} angles.")
        return hrtf_data
            
    def apply_hrtf_spatialization(
        self, audio: np.ndarray, width: float = 0.5
    ) -> np.ndarray:
        """
        Apply HRTF-based spatialization to create more immersive stereo.

        Args:
            audio: Input audio array (mono)
            width: Stereo width from 0.0 (mono) to 1.0 (maximum width)

        Returns:
            Spatialized stereo audio
        """
        if not self.use_hrtf or self.hrtf_data is None or len(audio.shape) > 1:
            # If HRTF is disabled, no data is available, or input is already stereo
            # Just apply basic stereo widening
            return self.apply_basic_stereo_widening(audio, width)

        # Create output stereo array
        stereo_out = np.zeros((len(audio), 2))

        # Set virtual source positions based on desired width
        # Full width would distribute sources across 180 degrees
        max_angle = 180 * width

        # Create 7 virtual sources at different angles
        angles = np.linspace(-max_angle / 2, max_angle / 2, 7)

        # Process each virtual source position
        for angle in angles:
            # Find closest angle in our HRTF dataset
            closest_angle = min(self.hrtf_data.keys(), key=lambda x: abs(x - angle))

            # Get the HRTF for this angle
            left_ir, right_ir = self.hrtf_data[closest_angle]

            # Apply HRTF convolution using FFT for efficiency
            left_channel = signal.fftconvolve(audio, left_ir, mode="same")
            right_channel = signal.fftconvolve(audio, right_ir, mode="same")

            # Add to output with position-dependent amplitude
            # (sources directly in front are louder)
            center_weight = 1.0 - (abs(angle) / (max_angle / 2))
            weight = 0.7 + 0.3 * center_weight  # Scale to [0.7, 1.0]

            stereo_out[:, 0] += left_channel * weight / len(angles)
            stereo_out[:, 1] += right_channel * weight / len(angles)

        # Normalize
        max_val = np.max(np.abs(stereo_out))
        if max_val > 0.95:
            stereo_out = stereo_out / max_val * 0.95

        return stereo_out

    def apply_basic_stereo_widening(
        self, audio: np.ndarray, width: float = 0.5
    ) -> np.ndarray:
        """
        Apply basic stereo widening without HRTF.

        Args:
            audio: Input audio array (mono or stereo)
            width: Stereo width from 0.0 (mono) to 1.0 (maximum width)

        Returns:
            Widened stereo audio
        """
        # Create output stereo array
        if len(audio.shape) == 1:
            # Mono input - convert to stereo
            stereo_out = np.zeros((len(audio), 2))

            # Delay-based widening
            delay_samples = int(0.0005 * self.sample_rate * width)  # 0.5ms max delay
            if delay_samples > 0:
                stereo_out[delay_samples:, 0] = audio[:-delay_samples]  # left delayed
                stereo_out[:-delay_samples, 1] = audio[delay_samples:]  # right delayed
            else:
                stereo_out[:, 0] = audio
                stereo_out[:, 1] = audio

            # Apply spectral differences for stereo width
            if width > 0.1:
                # Create subtle filtering for each ear
                b_left, a_left = signal.butter(
                    2, 5000 * (1 - width / 2) / (self.sample_rate / 2), "low"
                )
                b_right, a_right = signal.butter(
                    2, 5000 * (1 - width / 2) / (self.sample_rate / 2), "low"
                )

                # Apply filters
                stereo_out[:, 0] = signal.lfilter(b_left, a_left, stereo_out[:, 0])
                stereo_out[:, 1] = signal.lfilter(b_right, a_right, stereo_out[:, 1])

            # Add decorrelation for wider stereo image
            if width > 0.2:
                # Create decorrelated noise
                noise_len = int(0.01 * self.sample_rate)  # 10ms noise
                noise_left = np.random.randn(noise_len) * 0.01 * width
                noise_right = np.random.randn(noise_len) * 0.01 * width

                # Apply convolution
                left_decorr = signal.fftconvolve(stereo_out[:, 0], noise_left, mode="same")
                right_decorr = signal.fftconvolve(stereo_out[:, 1], noise_right, mode="same")

                # Mix with original
                stereo_out[:, 0] = stereo_out[:, 0] * 0.9 + left_decorr * 0.1
                stereo_out[:, 1] = stereo_out[:, 1] * 0.9 + right_decorr * 0.1
        else:
            # Already stereo - adjust width using mid/side processing
            mid = (audio[:, 0] + audio[:, 1]) / 2
            side = (audio[:, 0] - audio[:, 1]) / 2

            # Adjust width by scaling the side component
            side_scaled = side * width * 2  # Multiply by 2 for more pronounced effect

            # Recombine
            stereo_out = np.zeros_like(audio)
            stereo_out[:, 0] = mid + side_scaled
            stereo_out[:, 1] = mid - side_scaled

        # Normalize to prevent clipping
        max_val = np.max(np.abs(stereo_out))
        if max_val > 0.95:
            stereo_out = stereo_out / max_val * 0.95

        return stereo_out