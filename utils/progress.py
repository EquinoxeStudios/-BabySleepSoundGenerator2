"""
Progress tracking utilities for long-running operations.
"""

import time
import logging
from typing import Optional, Callable, Dict, Any

logger = logging.getLogger("BabySleepSoundGenerator")

class ProgressReporter:
    """
    Reports progress for long-running operations.
    Can report to both the logger and an optional callback function.
    """
    
    def __init__(self, total_steps: int, description: str = "Processing", callback: Optional[Callable] = None):
        """
        Initialize the progress reporter.
        
        Args:
            total_steps: Total number of steps in the operation
            description: Description of the operation
            callback: Optional callback function for progress updates, with signature:
                      callback(current_step, total_steps, percent, elapsed_seconds, eta_str, status_info)
        """
        self.total_steps = max(1, total_steps)  # Avoid division by zero
        self.current_step = 0
        self.description = description
        self.callback = callback
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.last_percent = 0
        self.status_info: Dict[str, Any] = {}
        
    def update(self, step: int = 1, force: bool = False, status_info: Optional[Dict[str, Any]] = None):
        """
        Update progress by specified number of steps.
        
        Args:
            step: Number of steps to advance
            force: Whether to force update even if threshold not met
            status_info: Optional status information to include in callback
        """
        self.current_step += step
        now = time.time()
        current_percent = min(100, int(100 * self.current_step / self.total_steps))
        
        # Update status info if provided
        if status_info:
            self.status_info.update(status_info)
        
        # Only update if percent changed or more than 1 second passed, or if forced
        if (force or 
            current_percent > self.last_percent or 
            (now - self.last_update_time) > 1.0):
            
            elapsed = now - self.start_time
            
            # Calculate ETA
            if self.current_step > 0 and self.current_step < self.total_steps:
                remaining_steps = self.total_steps - self.current_step
                step_time = elapsed / self.current_step
                eta = step_time * remaining_steps
                eta_str = f"ETA: {int(eta // 60)}m {int(eta % 60)}s"
            elif self.current_step >= self.total_steps:
                eta_str = "Complete"
            else:
                eta_str = "ETA: calculating..."
                
            # Log the progress
            if status_info and status_info.get('current_operation'):
                logger.info(f"{self.description}: {current_percent}% complete. "
                           f"[{status_info['current_operation']}] {eta_str}")
            else:
                logger.info(f"{self.description}: {current_percent}% complete. {eta_str}")
            
            # Call the callback if provided
            if self.callback:
                self.callback(self.current_step, self.total_steps, current_percent, 
                             elapsed, eta_str, self.status_info)
            
            # Update last values
            self.last_update_time = now
            self.last_percent = current_percent
    
    def complete(self, status_info: Optional[Dict[str, Any]] = None):
        """
        Mark the operation as complete.
        
        Args:
            status_info: Optional status information to include in callback
        """
        self.current_step = self.total_steps
        elapsed = time.time() - self.start_time
        
        # Update status info if provided
        if status_info:
            self.status_info.update(status_info)
        
        # Format elapsed time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        time_str = f"{minutes}m {seconds}s" if hours == 0 else f"{hours}h {minutes}m {seconds}s"
        
        # Log completion
        logger.info(f"{self.description}: 100% complete. Total time: {time_str}")
        
        # Call the callback if provided
        if self.callback:
            self.callback(self.total_steps, self.total_steps, 100, elapsed, 
                         f"Completed in {time_str}", self.status_info)