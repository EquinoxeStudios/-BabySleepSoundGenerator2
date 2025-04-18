# BabySleepSoundGenerator

A professional-quality audio generator for creating broadcast-standard baby sleep sounds with science-backed profiles.

![BabySleepSoundGenerator](https://img.shields.io/badge/Audio-Baby%20Sleep%20Sounds-blueviolet)
![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview

BabySleepSoundGenerator creates specialized white noise and ambient sounds for infants, designed with developmental science in mind. It produces broadcast-quality audio files suitable for YouTube channels, sleep aid applications, or personal use.

## Key Features

- **Research-Based Sound Profiles**: Tailored sound environments based on infant developmental stages
- **Broadcast-Standard Audio**: EBU R128 loudness normalization and ITU-R BS.1770 compliance
- **Advanced Sound Generation**:
  - FFT-based spectral shaping for accurate noise colors
  - Perlin noise for natural sound textures
  - Dynamic modulation to prevent listener fatigue
  - Cross-correlation for seamless looping
- **Sophisticated Processing**:
  - HRTF-based spatialization for immersive sound
  - Room acoustics simulation
  - Frequency contouring for optimal infant hearing
- **Safety Conscious**: Follows American Academy of Pediatrics guidelines for volume limits

## Sound Profiles

### Core Sound Types
- **White/Pink/Brown Noise**: Different colored noise profiles
- **Womb Sounds**: Authentic simulation of intrauterine acoustic environment
- **Heartbeat**: Variable or steady maternal-like heartbeat
- **Shushing**: Natural parental shushing sounds
- **Fan/Air**: Realistic fan and air conditioner sounds
- **Umbilical Swish**: Blood flow sounds from the womb

### Problem-Specific Profiles

| Profile | Description | Primary Elements |
|---------|-------------|------------------|
| **Newborn Transition** | Womb-like environment for adjusting to the outside world | Womb sounds, heartbeat, low frequencies |
| **Startle Reflex** | Consistent sound to prevent Moro reflex during sleep | Pink noise, reflective frequency patterns |
| **Colic Relief** | Louder, more intense profile for colic episodes | White noise, shushing, mid frequencies |
| **Sleep Regression (4-month)** | Balanced profile with circadian rhythm enhancements | Pink noise, circadian-aligned frequencies |
| **Teething Discomfort** | Distracting profile to help with teething pain | White noise, mid-high frequencies |
| **Toddler Resistance** | Calming low-frequency sound for winding down | Brown noise, low frequencies |

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/baby-sleep-sound-generator.git
   cd baby-sleep-sound-generator
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. For full functionality (recommended), install all optional dependencies:
   ```
   pip install noise pyloudnorm soundfile librosa pyroomacoustics pydub matplotlib numba tqdm
   ```

## Usage

### Command Line Interface

Generate a predefined profile for a specific sleep issue:
```
python main.py --mode predefined --problem newborn_transition --duration 8 --format wav
```

Create a custom sound with specific elements:
```
python main.py --mode custom --primary pink --overlay heartbeat shushing --focus low_mid --duration 10
```

See all available options:
```
python main.py --mode list
```

### Advanced Options

```
python main.py --mode custom --primary white --overlay heartbeat \
  --focus balanced --duration 8 --variable-heartbeat \
  --breathing-modulation --dynamic-volume --sample-rate 48000 --bit-depth 24
```

## Audio Quality Options

- **Sample rates**: 44100Hz, 48000Hz (default), 96000Hz
- **Bit depths**: 16-bit, 24-bit (default), 32-bit float
- **Output formats**: WAV (lossless), MP3 (compressed)

## Safety Guidelines

This generator adheres to the American Academy of Pediatrics guidelines:
- Maximum recommended volume: 65 dB SPL at 1 meter
- Placement recommendation: At least 7 feet (2.1m) from infant
- Auto-shutoff option available for extended-use safety

## Advanced Features

### Dynamic Effects

- **Variable Heartbeat**: Subtle heartbeat rate changes that mimic maternal variations
- **Breathing Modulation**: Gentle rhythm aligned with typical maternal breathing
- **Sleep Cycle Alignment**: Intensity modulation synchronized with infant sleep cycles
- **Moro Reflex Prevention**: Specialized sound patterns to reduce startle reflex
- **Dynamic Volume Reduction**: Gradual volume reduction to encourage deeper sleep over time

### Processing

- **Cross-correlation Looping**: Intelligent analysis to find optimal loop points
- **Spatial Widening**: HRTF-based spatialization for immersive sound environment
- **Room Acoustics**: Simulates different room sizes and acoustic properties
- **Frequency Contouring**: Precise spectral shaping for each developmental stage

## Project Structure

```
baby-sleep-sound-generator/
├── models/            # Data structures and constants
├── sound_profiles/    # Sound generators for different profiles
├── processing/        # Audio processing modules
├── effects/           # Special sound effects
├── output/            # Export and visualization
├── utils/             # Helper utilities
├── main.py            # Command-line interface
├── generator.py       # Core generator class
└── requirements.txt   # Dependencies
```

## Requirements

- Python 3.7+
- NumPy and SciPy (required)
- Additional libraries for enhanced functionality:
  - `noise`: For Perlin noise generation
  - `pyloudnorm`: For EBU R128 loudness normalization
  - `soundfile`: For advanced audio file handling
  - `librosa`: For enhanced audio processing
  - `pyroomacoustics`: For room simulation
  - `pydub`: For MP3 export (also requires ffmpeg)
  - `matplotlib`: For visualization
  - `numba`: For performance optimization

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Research on infant sleep and auditory development
- Professional audio engineering standards
- Digital signal processing techniques for natural sound generation