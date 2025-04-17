"""
Logging configuration for the BabySleepSoundGenerator.
"""

import logging


def setup_logging(verbose: bool = False) -> None:
    """
    Configure logging for the application.
    
    Args:
        verbose: Whether to use verbose logging level
    """
    # Set log level based on verbose flag
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Configure the root logger
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Get the logger for our application
    logger = logging.getLogger("BabySleepSoundGenerator")
    logger.setLevel(log_level)
    
    # Only add a new handler if there isn't one already to avoid duplicates
    if not logger.handlers:
        # Create a console handler
        handler = logging.StreamHandler()
        handler.setLevel(log_level)
        
        # Create a formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        
        # Add the handler to the logger
        logger.addHandler(handler)
    
    logger.info(f"Logging initialized at level: {'DEBUG' if verbose else 'INFO'}")