"""
Visualization of audio spectrum and characteristics.
"""

import os
import logging
import numpy as np
from scipy import signal
from typing import Optional

from utils.optional_imports import HAS_MATPLOTLIB, get_matplotlib_plt

logger = logging.getLogger("BabySleepSoundGenerator")


class SpectrumVisualizer:
    """Creates spectrum visualizations for audio files."""
    
    def __init__(self, sample_rate: int):
        """
        Initialize the spectrum visualizer.
        
        Args:
            sample_rate: Audio sample rate
        """
        self.sample_rate = sample_rate
        
    def visualize_spectrum(
        self, audio: np.ndarray, title: str, save_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Visualize the frequency spectrum of audio with key frequency markers.

        Args:
            audio: Audio array to analyze
            title: Title for the visualization
            save_path: Path to save the visualization image (optional)
            
        Returns:
            Path to the saved visualization file, or None if visualization failed
        """
        if not HAS_MATPLOTLIB:
            logger.warning("Matplotlib not installed. Visualization not available.")
            return None
            
        # Get matplotlib module
        plt = get_matplotlib_plt()
        if plt is None:
            logger.warning("Failed to get matplotlib.pyplot module.")
            return None
            
        # If no save path provided, create one from the title
        if save_path is None:
            save_path = f"{title.replace(' ', '_').lower()}_spectrum.png"
            
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

            # Define key frequencies of interest with labels
            key_freqs = {
                "20Hz": 20,
                "50Hz": 50,
                "100Hz": 100,
                "200Hz": 200,
                "500Hz": 500,
                "1kHz": 1000,
                "2kHz": 2000,
                "5kHz": 5000,
                "10kHz": 10000,
                "15kHz": 15000,
            }

            # Use matplotlib for visualization
            plt.figure(figsize=(12, 10))

            # Set up frequency analysis
            max_segment = 10 * self.sample_rate  # Analyze first 10 seconds max

            # Convert stereo to mono for analysis if needed
            if len(audio.shape) > 1:
                analysis_data = np.mean(audio[: min(len(audio), max_segment), :], axis=1)
            else:
                analysis_data = audio[: min(len(audio), max_segment)]

            # Compute power spectral density
            f, psd = signal.welch(analysis_data, self.sample_rate, nperseg=8192)

            # Convert to dB scale
            spectrum_db = 10 * np.log10(psd + 1e-10)

            # Plot the frequency spectrum
            plt.subplot(2, 1, 1)
            plt.semilogx(f, spectrum_db)
            plt.grid(True, which="both", ls="-", alpha=0.4)
            plt.xlim(20, self.sample_rate / 2)

            # Add reference lines for key frequencies
            for label, freq in key_freqs.items():
                if freq <= self.sample_rate / 2:
                    plt.axvline(x=freq, color="r", linestyle="--", alpha=0.3)
                    plt.text(
                        freq,
                        np.max(spectrum_db) - 5,
                        label,
                        horizontalalignment="center",
                        size="small",
                    )

            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Power Spectral Density (dB/Hz)")
            plt.title(f"{title} - Frequency Spectrum Analysis")

            # Add a spectrogram as second subplot
            plt.subplot(2, 1, 2)

            # Convert stereo to mono for spectrogram if needed
            if len(audio.shape) > 1:
                spectrogram_data = np.mean(audio[: min(len(audio), max_segment * 5), :], axis=1)
            else:
                spectrogram_data = audio[: min(len(audio), max_segment * 5)]

            f, t, Sxx = signal.spectrogram(
                spectrogram_data, self.sample_rate, nperseg=2048, noverlap=1024
            )
            plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading="gouraud")
            plt.yscale("log")
            plt.ylim(20, self.sample_rate / 2)
            plt.xlabel("Time (s)")
            plt.ylabel("Frequency (Hz)")
            plt.title("Spectrogram - Time-Frequency Analysis")
            plt.colorbar(label="Intensity (dB)")

            plt.tight_layout()

            # Save the figure
            plt.savefig(save_path, dpi=300)
            plt.close()
            
            logger.info(f"Spectrum visualization saved: {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"Error generating visualization: {e}")
            return None