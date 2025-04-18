"""
Sound configuration dataclass for comprehensive audio settings.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import os
import json
import logging

from models.constants import Constants, FrequencyFocus, RoomSize, OutputFormat

logger = logging.getLogger("BabySleepSoundGenerator")

@dataclass
class SoundConfiguration:
    """Comprehensive configuration for sound generation."""
    
    # Basic parameters
    name: str
    description: str = ""
    duration_seconds: float = 3600  # 1 hour default
    
    # Sound sources
    primary_sound: str = "white"  # primary sound type
    primary_sound_params: Dict[str, Any] = field(default_factory=dict)
    overlay_sounds: List[Dict[str, Any]] = field(default_factory=list)
    
    # Processing parameters
    sample_rate: int = Constants.DEFAULT_SAMPLE_RATE
    bit_depth: int = Constants.DEFAULT_BIT_DEPTH
    channels: int = Constants.DEFAULT_CHANNELS
    
    # Frequency shaping
    frequency_focus: str = FrequencyFocus.BALANCED.value
    frequency_emphasis: Optional[Dict[str, Any]] = None
    low_pass_filter: Optional[Dict[str, Any]] = None
    
    # Spatial processing
    spatial_width: float = 0.5
    use_hrtf: bool = True
    
    # Room acoustics
    room_size: str = RoomSize.MEDIUM.value
    room_simulation: bool = True
    
    # Effect processing
    breathing_modulation: Optional[Dict[str, Any]] = None
    sleep_cycle_modulation: Optional[Dict[str, Any]] = None
    dynamic_volume: Optional[Dict[str, Any]] = None
    moro_reflex_prevention: Optional[Dict[str, Any]] = None
    
    # Output parameters
    volume: float = 0.5
    target_loudness: float = Constants.DEFAULT_TARGET_LOUDNESS
    output_format: str = OutputFormat.WAV.value
    
    # Other
    seed: Optional[int] = None  # Random seed for reproducibility
    render_visualization: bool = False
    
    @classmethod
    def from_preset(cls, preset_name: str):
        """
        Create a configuration from a preset name.
        
        Args:
            preset_name: Name of the preset to load
            
        Returns:
            SoundConfiguration instance
        """
        # Load preset configurations
        from models.presets import get_preset
        preset_data = get_preset(preset_name)
        return cls(**preset_data)
    
    @classmethod
    def from_json(cls, json_file: str):
        """
        Create a configuration from a JSON file.
        
        Args:
            json_file: Path to JSON configuration file
            
        Returns:
            SoundConfiguration instance
        """
        if not os.path.exists(json_file):
            logger.warning(f"Configuration file not found: {json_file}")
            return cls(name=f"Default")
            
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            return cls(**data)
        except Exception as e:
            logger.error(f"Error loading configuration from {json_file}: {e}")
            return cls(name=f"Default (error loading {os.path.basename(json_file)})")
    
    def to_json(self, json_file: str):
        """
        Save configuration to a JSON file.
        
        Args:
            json_file: Path to save configuration
        """
        os.makedirs(os.path.dirname(json_file) or '.', exist_ok=True)
        with open(json_file, 'w') as f:
            json.dump(asdict(self), f, indent=2)
        logger.info(f"Configuration saved to {json_file}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def validate(self) -> bool:
        """
        Validate configuration.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        # TODO: Add comprehensive validation
        if self.duration_seconds <= 0:
            logger.error("Duration must be positive")
            return False
            
        if self.sample_rate <= 0:
            logger.error("Sample rate must be positive")
            return False
            
        if self.bit_depth not in [16, 24, 32]:
            logger.error("Bit depth must be 16, 24, or 32")
            return False
            
        if self.channels not in [1, 2]:
            logger.error("Channels must be 1 or 2")
            return False
            
        return True