"""
Configuration management for BabySleepSoundGenerator.
"""

import os
import json
import configparser
import logging
from typing import Any, Dict, Optional

from models.constants import Constants, HeartbeatConstants, ShushingConstants, FanConstants
from models.constants import WombConstants, UmbilicalConstants, PerformanceConstants, SpatialConstants

logger = logging.getLogger("BabySleepSoundGenerator")

class ConfigManager:
    """Manages configuration settings and provides access to constants."""
    
    _instance = None
    
    @classmethod
    def get_instance(cls, config_file=None):
        """Singleton access method."""
        if cls._instance is None:
            cls._instance = ConfigManager(config_file)
        elif config_file and os.path.exists(config_file):
            # If a new config is provided, reload
            cls._instance.load_config(config_file)
        return cls._instance
    
    def __init__(self, config_file=None):
        """Initialize with an optional config file."""
        self.config = configparser.ConfigParser()
        
        # Set default config
        self._set_defaults()
        
        # Load user config if provided
        if config_file:
            self.load_config(config_file)
    
    def _set_defaults(self):
        """Set default configuration values."""
        # General settings
        self.config["DEFAULT"] = {
            "sample_rate": str(Constants.DEFAULT_SAMPLE_RATE),
            "bit_depth": str(Constants.DEFAULT_BIT_DEPTH),
            "channels": str(Constants.DEFAULT_CHANNELS),
            "target_loudness": str(Constants.DEFAULT_TARGET_LOUDNESS),
            "crossfade_duration": str(Constants.DEFAULT_CROSSFADE_DURATION),
            "use_hrtf": "true",
            "room_simulation": "true",
        }
        
        # Section for heartbeat parameters
        self.config["HEARTBEAT"] = {
            "lub_duration": str(HeartbeatConstants.LUB_DURATION_SECONDS),
            "dub_duration": str(HeartbeatConstants.DUB_DURATION_SECONDS),
            "dub_delay": str(HeartbeatConstants.DUB_DELAY_SECONDS),
            "lub_frequency": str(HeartbeatConstants.LUB_FREQUENCY_HZ),
            "dub_frequency": str(HeartbeatConstants.DUB_FREQUENCY_HZ),
            "amplitude_variation_pct": str(HeartbeatConstants.AMPLITUDE_VARIATION_PCT),
            "default_amplitude": str(HeartbeatConstants.DEFAULT_AMPLITUDE),
        }
        
        # Section for shushing parameters
        self.config["SHUSHING"] = {
            "bandpass_low_hz": str(ShushingConstants.BANDPASS_LOW_HZ),
            "bandpass_high_hz": str(ShushingConstants.BANDPASS_HIGH_HZ),
            "shush_rate_per_second": str(ShushingConstants.SHUSH_RATE_PER_SECOND),
            "modulation_min": str(ShushingConstants.MODULATION_MIN),
            "modulation_max": str(ShushingConstants.MODULATION_MAX),
            "default_amplitude": str(ShushingConstants.DEFAULT_AMPLITUDE),
        }
        
        # Section for fan parameters
        self.config["FAN"] = {
            "base_rotation_hz": str(FanConstants.BASE_ROTATION_HZ),
            "speed_variation_pct": str(FanConstants.SPEED_VARIATION_PCT),
            "resonance_q_factor": str(FanConstants.RESONANCE_Q_FACTOR),
            "bandpass_low_hz": str(FanConstants.BANDPASS_LOW_HZ),
            "bandpass_high_hz": str(FanConstants.BANDPASS_HIGH_HZ),
            "default_amplitude": str(FanConstants.DEFAULT_AMPLITUDE),
        }
        
        # Section for womb parameters
        self.config["WOMB"] = {
            "breathing_rate_bpm": str(WombConstants.BREATHING_RATE_BREATHS_PER_MIN),
            "blood_flow_cpm": str(WombConstants.BLOOD_FLOW_CYCLES_PER_MIN),
            "breathing_modulation_depth": str(WombConstants.BREATHING_MODULATION_DEPTH),
            "deep_rhythm_modulation_depth": str(WombConstants.DEEP_RHYTHM_MODULATION_DEPTH),
            "lowpass_cutoff_hz": str(WombConstants.WOMB_LOWPASS_CUTOFF_HZ),
            "default_amplitude": str(WombConstants.DEFAULT_AMPLITUDE),
        }
        
        # Section for umbilical parameters
        self.config["UMBILICAL"] = {
            "whoosh_bandpass_low_hz": str(UmbilicalConstants.WHOOSH_BANDPASS_LOW_HZ),
            "whoosh_bandpass_high_hz": str(UmbilicalConstants.WHOOSH_BANDPASS_HIGH_HZ),
            "pulse_rate_bpm": str(UmbilicalConstants.PULSE_RATE_BPM),
            "pulse_factor_base": str(UmbilicalConstants.PULSE_FACTOR_BASE),
            "pulse_factor_variation": str(UmbilicalConstants.PULSE_FACTOR_VARIATION),
            "default_amplitude": str(UmbilicalConstants.DEFAULT_AMPLITUDE),
        }
        
        # Section for performance parameters
        self.config["PERFORMANCE"] = {
            "buffer_size": str(PerformanceConstants.DEFAULT_BUFFER_SIZE),
            "perlin_stretch_factor": str(PerformanceConstants.PERLIN_STRETCH_FACTOR),
            "fft_chunk_size_seconds": str(PerformanceConstants.FFT_CHUNK_SIZE_SECONDS),
            "max_duration_before_chunking": str(PerformanceConstants.MAX_DURATION_SECONDS_BEFORE_CHUNKING),
            "crossfade_between_chunks": str(PerformanceConstants.CROSSFADE_BETWEEN_CHUNKS_SECONDS),
        }
        
        # Section for spatial parameters
        self.config["SPATIAL"] = {
            "max_stereo_delay_ms": str(SpatialConstants.MAX_STEREO_DELAY_MS),
            "hrtf_impulse_length": str(SpatialConstants.HRTF_IMPULSE_LENGTH),
            "decorrelation_noise_duration_ms": str(SpatialConstants.DECORRELATION_NOISE_DURATION_MS),
            "default_stereo_width": str(SpatialConstants.DEFAULT_STEREO_WIDTH),
        }
    
    def load_config(self, config_file):
        """Load configuration from a file."""
        if os.path.exists(config_file):
            logger.info(f"Loading configuration from {config_file}")
            try:
                self.config.read(config_file)
                logger.info("Configuration loaded successfully")
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")
        else:
            logger.warning(f"Configuration file {config_file} not found. Using defaults.")
    
    def get_int(self, section, key, default=None):
        """Get an integer value from config."""
        try:
            return self.config.getint(section, key)
        except (configparser.NoSectionError, configparser.NoOptionError, ValueError):
            return default
    
    def get_float(self, section, key, default=None):
        """Get a float value from config."""
        try:
            return self.config.getfloat(section, key)
        except (configparser.NoSectionError, configparser.NoOptionError, ValueError):
            return default
    
    def get_bool(self, section, key, default=None):
        """Get a boolean value from config."""
        try:
            return self.config.getboolean(section, key)
        except (configparser.NoSectionError, configparser.NoOptionError, ValueError):
            return default
    
    def get_str(self, section, key, default=None):
        """Get a string value from config."""
        try:
            return self.config.get(section, key)
        except (configparser.NoSectionError, configparser.NoOptionError):
            return default
    
    def get_list(self, section, key, default=None, item_type=str):
        """Get a list value from config."""
        try:
            value = self.config.get(section, key)
            items = [x.strip() for x in value.split(',')]
            
            # Convert items to the specified type
            if item_type == int:
                return [int(x) for x in items]
            elif item_type == float:
                return [float(x) for x in items]
            elif item_type == bool:
                return [x.lower() in ('true', 'yes', '1', 'y') for x in items]
            else:
                return items
        except (configparser.NoSectionError, configparser.NoOptionError, ValueError):
            return default if default is not None else []
    
    def set(self, section, key, value):
        """Set a value in the configuration."""
        if not self.config.has_section(section) and section != 'DEFAULT':
            self.config.add_section(section)
        self.config.set(section, key, str(value))
    
    def save(self, config_file):
        """Save the current configuration to a file."""
        os.makedirs(os.path.dirname(config_file) or '.', exist_ok=True)
        with open(config_file, 'w') as f:
            self.config.write(f)
        logger.info(f"Configuration saved to {config_file}")
        
    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """Convert configuration to nested dictionary."""
        result = {}
        for section in self.config.sections():
            result[section] = {}
            for key, value in self.config.items(section):
                result[section][key] = value
        
        # Add DEFAULT section
        result['DEFAULT'] = {}
        for key, value in self.config.items('DEFAULT'):
            result['DEFAULT'][key] = value
            
        return result
        
    def export_json(self, json_file):
        """Export configuration to JSON file."""
        config_dict = self.to_dict()
        os.makedirs(os.path.dirname(json_file) or '.', exist_ok=True)
        with open(json_file, 'w') as f:
            json.dump(config_dict, f, indent=2)
        logger.info(f"Configuration exported to JSON: {json_file}")
    
    @classmethod
    def from_json(cls, json_file):
        """Create a ConfigManager from a JSON file."""
        config_manager = cls()
        
        if os.path.exists(json_file):
            try:
                with open(json_file, 'r') as f:
                    config_dict = json.load(f)
                
                # Clear existing config and load from dict
                config_manager.config = configparser.ConfigParser()
                
                for section, section_dict in config_dict.items():
                    if section != 'DEFAULT':
                        config_manager.config.add_section(section)
                    
                    for key, value in section_dict.items():
                        config_manager.config.set(section, key, str(value))
                
                logger.info(f"Configuration loaded from JSON: {json_file}")
            except Exception as e:
                logger.error(f"Error loading JSON configuration: {e}")
        else:
            logger.warning(f"JSON configuration file {json_file} not found. Using defaults.")
        
        return config_manager