"""
Parallel processing utilities for efficient computation.
"""

import concurrent.futures
import multiprocessing  # Added this import for cpu_count
import logging
import time
from typing import List, Callable, Any, Dict, Tuple, Optional

logger = logging.getLogger("BabySleepSoundGenerator")

class ParallelProcessor:
    """
    Handles parallel processing of tasks.
    """
    
    def __init__(self, max_workers: Optional[int] = None):
        """
        Initialize the parallel processor.
        
        Args:
            max_workers: Maximum number of worker threads/processes
        """
        self.max_workers = max_workers
        
    def process(self, 
                tasks: List[Tuple[Callable, Dict[str, Any]]], 
                use_threads: bool = True,
                description: str = "Processing tasks") -> List[Any]:
        """
        Process tasks in parallel.
        
        Args:
            tasks: List of (function, kwargs) tuples to execute
            use_threads: Whether to use thread pool (True) or process pool (False)
            description: Description for logging
            
        Returns:
            List of results in the same order as tasks
        """
        if not tasks:
            logger.warning("No tasks provided to ParallelProcessor")
            return []
            
        results = []
        num_tasks = len(tasks)
        executor_cls = concurrent.futures.ThreadPoolExecutor if use_threads else concurrent.futures.ProcessPoolExecutor
        
        # Calculate max workers - default is min(tasks, CPU count x 5 for threads, CPU count for processes)
        cpu_multiplier = 5 if use_threads else 1
        workers = self.max_workers or min(num_tasks, multiprocessing.cpu_count() * cpu_multiplier)  # Fixed this line
        
        logger.info(f"Starting {num_tasks} parallel tasks using {workers} workers ({description})")
        start_time = time.time()
        
        with executor_cls(max_workers=workers) as executor:
            # Start all tasks
            future_to_idx = {
                executor.submit(func, **kwargs): i
                for i, (func, kwargs) in enumerate(tasks)
            }
            
            # Prepare results list with placeholders
            results = [None] * num_tasks
            
            # Collect results as they complete
            completed = 0
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                completed += 1
                
                try:
                    result = future.result()
                    # Store at correct position to maintain order
                    results[idx] = result
                    logger.debug(f"Task {idx+1}/{num_tasks} completed: {completed}/{num_tasks} done")
                except Exception as e:
                    logger.error(f"Task {idx+1}/{num_tasks} failed: {e}")
                    results[idx] = None
                
                # Periodically report progress for long tasks
                if completed % max(1, num_tasks // 10) == 0 or completed == num_tasks:
                    elapsed = time.time() - start_time
                    logger.info(f"{description}: {completed}/{num_tasks} tasks completed "
                               f"({int(100 * completed / num_tasks)}%) in {elapsed:.2f}s")
        
        total_time = time.time() - start_time
        logger.info(f"All {num_tasks} tasks completed in {total_time:.2f}s ({description})")
        return results
    
    def process_batched(self,
                       items: List[Any],
                       process_func: Callable,
                       batch_size: int = 10,
                       use_threads: bool = True,
                       description: str = "Processing items") -> List[Any]:
        """
        Process a list of items in batches.
        
        Args:
            items: List of items to process
            process_func: Function to process each item, with signature: func(item, **kwargs)
            batch_size: Number of items to process in each batch
            use_threads: Whether to use thread pool (True) or process pool (False)
            description: Description for logging
            
        Returns:
            List of results in the same order as items
        """
        if not items:
            logger.warning("No items provided to process_batched")
            return []
            
        total_items = len(items)
        num_batches = (total_items + batch_size - 1) // batch_size
        results = []
        
        logger.info(f"Processing {total_items} items in {num_batches} batches (batch size: {batch_size})")
        start_time = time.time()
        
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, total_items)
            batch_items = items[batch_start:batch_end]
            
            batch_description = f"{description} - Batch {batch_idx+1}/{num_batches}"
            logger.info(f"Processing {len(batch_items)} items in {batch_description}")
            
            # Prepare tasks for this batch
            tasks = [
                (process_func, {"item": item})
                for item in batch_items
            ]
            
            # Process this batch
            batch_results = self.process(tasks, use_threads, batch_description)
            results.extend(batch_results)
            
            # Report progress
            processed = batch_end
            elapsed = time.time() - start_time
            logger.info(f"{description}: {processed}/{total_items} items processed "
                       f"({int(100 * processed / total_items)}%) in {elapsed:.2f}s")
        
        total_time = time.time() - start_time
        logger.info(f"All {total_items} items processed in {total_time:.2f}s ({description})")
        return results