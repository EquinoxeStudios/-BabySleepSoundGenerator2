"""
Room acoustics simulation and processing.
"""

import logging
import numpy as np
from scipy import signal

from models.constants import RoomSize, Constants
from utils.optional_imports import HAS_PYROOMACOUSTICS

logger = logging.getLogger("BabySleepSoundGenerator")

# Import optional libraries
if HAS_PYROOMACOUSTICS:
    import pyroomacoustics as pra


class RoomAcousticsProcessor:
    """Processor for room acoustics simulation."""
    
    def __init__(self, sample_rate: int, room_simulation: bool = True):
        """
        Initialize the room acoustics processor.
        
        Args:
            sample_rate: Audio sample rate
            room_simulation: Whether to use room acoustics simulation
        """
        self.sample_rate = sample_rate
        self.room_simulation = room_simulation and HAS_PYROOMACOUSTICS
        
        # Load room impulse response data if requested
        self.room_ir = None
        if self.room_simulation:
            try:
                self.room_ir = self._generate_room_impulse_response()
            except Exception as e:
                logger.warning(f"Could not generate room impulse response: {e}")
                self.room_simulation = False
        
    def _generate_room_impulse_response(
        self, room_size: RoomSize = RoomSize.MEDIUM, rt60: float = 0.3
    ) -> np.ndarray:
        """
        Generate room impulse response for a given room size and reverberation time.

        Args:
            room_size: Small, medium, or large room
            rt60: Reverberation time (seconds)

        Returns:
            Room impulse response array for convolution
        """
        logger.info(f"Generating room impulse response for {room_size} room...")
        
        if not HAS_PYROOMACOUSTICS:
            # Create a simple placeholder impulse if the library isn't available
            ir_length = int(rt60 * self.sample_rate)
            ir = np.zeros(ir_length)
            ir[0] = 1.0  # Direct sound

            # Add some early reflections
            for i in range(5):
                pos = int((i + 1) * 0.01 * self.sample_rate)  # reflections at 10ms intervals
                if pos < ir_length:
                    ir[pos] = 0.7 * (0.7**i)  # Decaying reflections

            # Add exponential decay for late reverb
            t = np.arange(ir_length) / self.sample_rate
            ir += 0.2 * np.exp(-5 * t) * np.random.randn(ir_length)

            # Normalize
            ir = ir / np.max(np.abs(ir))

            # Create stereo IR
            ir_stereo = np.zeros((ir_length, 2))
            ir_stereo[:, 0] = ir
            ir_stereo[:, 1] = ir

            return ir_stereo

        # Set room dimensions based on room size
        if room_size == RoomSize.SMALL:
            room_dim = [3, 4, 2.5]  # Small bedroom/nursery in meters
        elif room_size == RoomSize.MEDIUM:
            room_dim = [5, 6, 2.7]  # Medium room
        else:  # large
            room_dim = [8, 10, 3.0]  # Large room

        # Create the room
        room = pra.ShoeBox(
            room_dim,
            fs=self.sample_rate,
            materials=pra.Material(0.2, 0.15),  # Material absorption coefficients
            max_order=15,
        )  # Number of reflections to compute

        # Add a source somewhere in the room
        source_pos = [room_dim[0] / 2, room_dim[1] / 2, 1.5]  # Center of room, speaker height
        room.add_source(source_pos)

        # Add microphone array (stereo pair)
        mic_distance = 0.2  # 20 cm between mics for stereo
        mic_pos = np.c_[
            [room_dim[0] / 2 - mic_distance / 2, room_dim[1] / 2, 1.0],  # Left mic
            [room_dim[0] / 2 + mic_distance / 2, room_dim[1] / 2, 1.0],  # Right mic
        ]
        room.add_microphone_array(mic_pos)

        # Compute the room impulse response
        room.compute_rir()

        # Get the stereo impulse response
        rir_left = room.rir[0][0]
        rir_right = room.rir[1][0]

        # Make sure both channels have the same length
        max_length = max(len(rir_left), len(rir_right))
        ir_stereo = np.zeros((max_length, 2))

        ir_stereo[: len(rir_left), 0] = rir_left
        ir_stereo[: len(rir_right), 1] = rir_right

        # Trim to a reasonable length to save computation
        max_ir_length = int(rt60 * 2 * self.sample_rate)  # Twice the RT60 time
        if len(ir_stereo) > max_ir_length:
            ir_stereo = ir_stereo[:max_ir_length, :]

        logger.info(f"Room impulse response generated: {ir_stereo.shape}")
        return ir_stereo

    def apply_room_acoustics(
        self, audio: np.ndarray, room_size: RoomSize = RoomSize.MEDIUM
    ) -> np.ndarray:
        """
        Apply room acoustics simulation to the audio.

        Args:
            audio: Input audio array (can be mono or stereo)
            room_size: Room size for impulse response

        Returns:
            Audio with room acoustics applied
        """
        if not self.room_simulation:
            return audio

        # Generate room impulse response if needed
        if self.room_ir is None or len(self.room_ir) == 0:
            self.room_ir = self._generate_room_impulse_response(room_size)

        # Convert mono to stereo if needed
        if len(audio.shape) == 1:
            audio_stereo = np.zeros((len(audio), 2))
            audio_stereo[:, 0] = audio
            audio_stereo[:, 1] = audio
        else:
            audio_stereo = audio

        # Apply convolution for each channel using FFT convolution for efficiency
        reverb_left = signal.fftconvolve(audio_stereo[:, 0], self.room_ir[:, 0], mode="full")
        reverb_right = signal.fftconvolve(audio_stereo[:, 1], self.room_ir[:, 1], mode="full")

        # Trim to original length
        reverb_left = reverb_left[: len(audio)]
        reverb_right = reverb_right[: len(audio)]

        # Combine channels
        reverb = np.zeros((len(audio), 2))
        reverb[:, 0] = reverb_left
        reverb[:, 1] = reverb_right

        # Mix with dry signal for more control: 80% wet (reverb), 20% dry (original)
        mixed = 0.8 * reverb + 0.2 * audio_stereo

        # Normalize to prevent clipping
        max_val = np.max(np.abs(mixed))
        if max_val > Constants.MAX_AUDIO_VALUE:
            mixed = mixed / max_val * Constants.MAX_AUDIO_VALUE

        return mixed