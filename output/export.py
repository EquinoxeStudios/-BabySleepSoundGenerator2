"""
Audio file export functionality (WAV and MP3).
"""

import os
import logging
import tempfile
import wave
import struct
import numpy as np
from scipy.io import wavfile

from utils.optional_imports import HAS_SOUNDFILE, HAS_PYDUB

logger = logging.getLogger("BabySleepSoundGenerator")

# Import optional libraries
if HAS_SOUNDFILE:
    import soundfile as sf

if HAS_PYDUB:
    from pydub import AudioSegment


class AudioExporter:
    """Handles exporting audio to various file formats."""
    
    def __init__(self, sample_rate: int, bit_depth: int, channels: int):
        """
        Initialize the audio exporter.
        
        Args:
            sample_rate: Audio sample rate
            bit_depth: Bit depth (16, 24, or 32)
            channels: Number of channels (1 or 2)
        """
        self.sample_rate = sample_rate
        self.bit_depth = bit_depth
        self.channels = channels
        
    def save_to_wav(self, audio: np.ndarray, filename: str, volume: float = 1.0) -> str:
        """
        Save the audio array to a WAV file with applied volume and correct bit depth
        
        Args:
            audio: Audio array to save
            filename: Output filename
            volume: Volume level (0.0-1.0)
            
        Returns:
            Path to the saved file
        """
        # Ensure filename has .wav extension
        if not filename.lower().endswith('.wav'):
            filename = f"{os.path.splitext(filename)[0]}.wav"
            
        # Apply simple volume adjustment
        audio = audio * volume

        # Make sure we don't exceed [-1, 1] range
        max_val = np.max(np.abs(audio))
        if max_val > 1.0:
            audio = audio / max_val

        # Ensure directory exists
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)

        # Save to WAV with appropriate bit depth
        if HAS_SOUNDFILE:
            # Use soundfile for better control over audio format
            if self.bit_depth == 16:
                subtype = "PCM_16"
            elif self.bit_depth == 24:
                subtype = "PCM_24"
            else:
                subtype = "FLOAT"

            # Ensure audio is in the right shape for soundfile
            if len(audio.shape) == 1 and self.channels == 2:
                # Convert mono to stereo
                stereo_audio = np.zeros((len(audio), 2))
                stereo_audio[:, 0] = audio
                stereo_audio[:, 1] = audio
                sf.write(filename, stereo_audio, self.sample_rate, subtype=subtype)
            else:
                sf.write(filename, audio, self.sample_rate, subtype=subtype)
        else:
            # Fall back to scipy and wave
            if self.bit_depth == 16:
                # 16-bit WAV
                if len(audio.shape) == 1 and self.channels == 2:
                    # Convert mono to stereo
                    stereo_audio = np.zeros((len(audio), 2))
                    stereo_audio[:, 0] = audio
                    stereo_audio[:, 1] = audio
                    audio_as_int = (stereo_audio * 32767).astype(np.int16)
                else:
                    audio_as_int = (audio * 32767).astype(np.int16)
                wavfile.write(filename, self.sample_rate, audio_as_int)
            elif self.bit_depth == 24:
                # 24-bit WAV requires a bit more work
                with wave.open(filename, "w") as wav_file:
                    wav_file.setnchannels(self.channels)
                    wav_file.setsampwidth(3)  # 3 bytes for 24-bit
                    wav_file.setframerate(self.sample_rate)

                    # Convert to 24-bit PCM
                    if len(audio.shape) == 1:
                        # Mono
                        if self.channels == 2:
                            # Duplicate for stereo
                            audio_as_int = (audio.reshape(-1, 1) * 8388607).astype(np.int32)
                            audio_as_int = np.column_stack((audio_as_int, audio_as_int))
                        else:
                            audio_as_int = (audio * 8388607).astype(np.int32)
                    else:
                        # Already multi-channel
                        audio_as_int = (audio * 8388607).astype(np.int32)

                    # Pack as bytes
                    if self.channels == 2:
                        # Stereo
                        for i in range(len(audio_as_int)):
                            # Convert to 3 bytes in little-endian format for each channel
                            left_bytes = audio_as_int[i, 0].tobytes()[:3]
                            right_bytes = audio_as_int[i, 1].tobytes()[:3]
                            wav_file.writeframesraw(left_bytes + right_bytes)
                    else:
                        # Mono
                        for sample in audio_as_int:
                            # Convert to 3 bytes in little-endian format
                            bytes_data = sample.tobytes()[:3]
                            wav_file.writeframesraw(bytes_data)
            else:
                # 32-bit float WAV
                if len(audio.shape) == 1 and self.channels == 2:
                    # Convert mono to stereo
                    stereo_audio = np.zeros((len(audio), 2))
                    stereo_audio[:, 0] = audio
                    stereo_audio[:, 1] = audio
                    wavfile.write(filename, self.sample_rate, stereo_audio.astype(np.float32))
                else:
                    wavfile.write(filename, self.sample_rate, audio.astype(np.float32))

        logger.info(f"WAV file saved: {filename}")
        return filename

    def save_to_mp3(self, audio: np.ndarray, filename: str, bitrate: str = "320k") -> str:
        """
        Save the audio to an MP3 file (requires pydub and ffmpeg).
    
        Args:
             audio: The audio array
             filename: Output filename
             bitrate: MP3 bitrate (default: 320k for highest quality)
        
        Returns:
             Path to the created file
        """
        # Ensure filename has .mp3 extension
        if not filename.lower().endswith('.mp3'):
            filename = f"{os.path.splitext(filename)[0]}.mp3"
        
        # Create a temporary file for the WAV
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_wav = temp_file.name
        
        # Save as temporary WAV
        self.save_to_wav(audio, temp_wav)
        
        # Check if pydub is available and ffmpeg is accessible
        if not HAS_PYDUB:
            logger.warning("Pydub not installed or ffmpeg not found. Falling back to WAV format.")
            # If MP3 export fails, return the WAV file
            wav_filename = os.path.splitext(filename)[0] + ".wav"
            os.rename(temp_wav, wav_filename)
            return wav_filename
        
        try:
            # Load the audio using pydub
            audio_segment = AudioSegment.from_wav(temp_wav)
            
            # Export as MP3 with broadcast-standard tags
            tags = {
                'title': os.path.splitext(os.path.basename(filename))[0],
                'artist': 'BabySleepSoundGenerator',
                'album': 'Sleep Sounds for Babies',
                'comment': 'Created with BabySleepSoundGenerator',
                'genre': 'Ambient'
            }
            
            # Make sure directory exists
            os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)
            
            # Export with tags and bitrate
            audio_segment.export(
                filename,
                format="mp3",
                bitrate=bitrate,
                tags=tags,
                parameters=["-q:a", "0"]  # Use highest quality settings
            )
            
            # Remove temporary WAV file
            try:
                os.remove(temp_wav)
            except OSError:
                logger.warning(f"Could not remove temporary file {temp_wav}")
                
            logger.info(f"MP3 file saved: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error exporting to MP3: {e}")
            logger.warning("Falling back to WAV format")
            
            # If MP3 export fails, return the WAV file
            wav_filename = os.path.splitext(filename)[0] + ".wav"
            os.rename(temp_wav, wav_filename)
            logger.info(f"WAV file saved: {wav_filename}")
            return wav_filename