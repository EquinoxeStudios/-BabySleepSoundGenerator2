"""
Logging configuration for the BabySleepSoundGenerator.
"""

import logging
import logging.handlers
import os
import sys
import time
from typing import Optional, Dict, Any, Union

# Singleton pattern to manage logging configuration state
class LoggingManager:
    """Singleton manager for logging configuration to prevent duplicate handlers."""
    _instance = None
    _configured_loggers = set()
    
    @classmethod
    def get_instance(cls):
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = LoggingManager()
        return cls._instance
    
    def is_logger_configured(self, logger_name: str) -> bool:
        """Check if a logger has already been configured."""
        return logger_name in self._configured_loggers
    
    def mark_logger_configured(self, logger_name: str) -> None:
        """Mark a logger as configured."""
        self._configured_loggers.add(logger_name)


def setup_logging(
    verbose: bool = False,
    log_to_file: bool = False,
    log_file: Optional[str] = None,
    log_level: Optional[Union[int, str]] = None,
    log_format: Optional[str] = None,
    enable_rotation: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
    add_context: Optional[Dict[str, Any]] = None
) -> logging.Logger:
    """
    Configure logging for the application with enhanced features.
    
    Args:
        verbose: Whether to use verbose logging level
        log_to_file: Whether to log to a file
        log_file: Path to log file (defaults to babysleepsound_{timestamp}.log in cwd)
        log_level: Custom log level (overrides verbose flag)
        log_format: Custom log format string
        enable_rotation: Whether to enable log rotation
        max_bytes: Maximum log file size before rotation
        backup_count: Number of backup files to keep
        add_context: Additional context information to include in log records
        
    Returns:
        Configured logger object
    """
    # Get the logging manager singleton
    manager = LoggingManager.get_instance()
    
    # Get the logger for our application
    logger = logging.getLogger("BabySleepSoundGenerator")
    
    # Skip configuration if already configured
    if manager.is_logger_configured(logger.name):
        return logger
    
    # Set log level based on parameters
    if log_level is not None:
        if isinstance(log_level, str):
            level = getattr(logging, log_level.upper(), logging.INFO)
        else:
            level = log_level
    else:
        level = logging.DEBUG if verbose else logging.INFO
    
    # Set the logger level
    logger.setLevel(level)
    
    # Default log format with more context
    default_format = '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(threadName)s - %(message)s'
    
    # Use custom format if provided
    format_str = log_format or default_format
    
    # Create formatter
    formatter = logging.Formatter(format_str)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Add file handler if requested
    if log_to_file:
        # Create log directory if it doesn't exist
        if log_file:
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
        else:
            # Default log file in current directory
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            log_file = f"babysleepsound_{timestamp}.log"
        
        if enable_rotation:
            # Use rotating file handler
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count
            )
        else:
            # Use standard file handler
            file_handler = logging.FileHandler(log_file)
            
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Add context filter if provided
    if add_context:
        class ContextFilter(logging.Filter):
            def filter(self, record):
                for key, value in add_context.items():
                    setattr(record, key, value)
                return True
        
        context_filter = ContextFilter()
        logger.addFilter(context_filter)
    
    # Mark this logger as configured
    manager.mark_logger_configured(logger.name)
    
    logger.info(f"Logging initialized at level: {logging.getLevelName(level)}")
    if log_to_file:
        logger.info(f"Logging to file: {log_file}" + 
                   (f" with rotation ({max_bytes/1024/1024:.1f}MB, {backup_count} backups)" 
                    if enable_rotation else ""))
    
    return logger