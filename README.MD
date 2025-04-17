# BabySleepSoundGenerator

A professional-quality audio generator for creating customized baby sleep sounds with broadcast-standard quality.

## Features

- **High-quality Sound Generation**: Uses advanced techniques like FFT-based spectral shaping and Perlin noise for natural sound textures
- **Professionally-tuned Sound Profiles**: Research-backed profiles for specific sleep issues
- **Broadcast-standard Audio**: EBU R128 loudness normalization and ITU-R BS.1770 compliance
- **Sophisticated Processing**: HRTF-based spatialization, room acoustics simulation, and more
- **Developmental Science**: Sound profiles based on baby developmental stages

## Sound Profile Options

- **White/Pink/Brown Noise**: Different colored noise profiles
- **Womb Sounds**: Recreates intrauterine acoustic environment
- **Heartbeat**: Variable or steady maternal-like heartbeat
- **Shushing**: Natural parental shushing sounds
- **Fan/Air**: Realistic fan and air conditioner sounds
- **Umbilical Swish**: Blood flow sounds from the womb

## Predefined Problem Profiles

- **Newborn Transition**: Womb-like environment for adjusting to the outside world
- **Startle Reflex**: Consistent sound to prevent Moro reflex during sleep
- **Colic Relief**: Louder, more intense profile for colic episodes
- **Sleep Regression (4-month)**: Balanced profile with circadian rhythm enhancements
- **Teething Discomfort**: Distracting profile to help with teething pain
- **Toddler Resistance**: Calming low-frequency sound for winding down

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

3. For full functionality, install optional dependencies:
   ```
   pip install noise pyloudnorm soundfile librosa pyroomacoustics pydub matplotlib numba
   ```

## Usage

### Command Line Interface

Generate a predefined profile:
```
python main.py --mode predefined --problem newborn_transition --duration 8 --format wav
```

Create a custom sound:
```
python main.py --mode custom --primary pink --overlay heartbeat --focus low_mid --duration 10
```

List available options:
```
python main.py --mode list
```

### Advanced Options

```
python main.py --mode custom --primary white --overlay shushing --focus balanced --duration 8 --variable-heartbeat --breathing-modulation --dynamic-volume --sample-rate 48000 --bit-depth 24
```

## Audio Quality Options

- Sample rates: 44100Hz, 48000Hz (default), 96000Hz
- Bit depths: 16-bit, 24-bit (default), 32-bit float
- Output formats: WAV (lossless), MP3 (compressed)

## Safety Guidelines

This generator follows the American Academy of Pediatrics guidelines:
- Maximum recommended volume: 65 dB SPL at 1 meter
- Placement recommendation: At least 7 feet (2.1m) from infant
- Auto-shutoff option available for safety

## Development

The project is organized into modules:
- `sound_profiles/`: Different sound generators
- `processing/`: Audio processing modules
- `effects/`: Special sound effects
- `output/`: Export and visualization
- `models/`: Data structures and constants
- `utils/`: Helper utilities

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Research on infant sleep and auditory development
- Professional audio engineering standards
- Advanced DSP techniques for natural sound generation