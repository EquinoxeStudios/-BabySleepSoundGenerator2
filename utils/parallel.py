"""
Parallel processing utilities for efficient computation.
"""

import concurrent.futures
import multiprocessing
import logging
import time
import os
import threading
import psutil
from typing import List, Callable, Any, Dict, Tuple, Optional, Union, TypeVar, Generic

logger = logging.getLogger("BabySleepSoundGenerator")

# Type variables for generic functions
T = TypeVar('T')
R = TypeVar('R')

class TaskResult(Generic[R]):
    """
    Wrapper for task execution results with error information.
    """
    def __init__(
        self, 
        result: Optional[R] = None, 
        error: Optional[Exception] = None,
        task_id: Optional[int] = None,
        duration: float = 0.0
    ):
        self.result = result
        self.error = error
        self.task_id = task_id
        self.duration = duration
        self.success = error is None
    
    def __bool__(self) -> bool:
        """Returns True if execution was successful."""
        return self.success
    
    def unwrap(self) -> R:
        """
        Unwrap the result or raise the stored exception.
        
        Returns:
            The result value if successful
            
        Raises:
            The stored exception if not successful
        """
        if self.error is not None:
            raise self.error
        return self.result


class ProgressTracker:
    """
    Thread-safe progress tracking for parallel tasks.
    """
    def __init__(self, total: int, desc: str = "Processing", log_interval: float = 1.0):
        self.total = max(1, total)  # Avoid division by zero
        self.desc = desc
        self.log_interval = log_interval
        self.completed = 0
        self.start_time = time.time()
        self.last_log_time = self.start_time
        self._lock = threading.Lock()
    
    def increment(self, count: int = 1) -> None:
        """
        Increment the completed count and log progress if needed.
        
        Args:
            count: Number of items to increment by
        """
        with self._lock:
            self.completed += count
            now = time.time()
            
            # Only log if enough time has passed since last log
            if now - self.last_log_time >= self.log_interval:
                self._log_progress(now)
                self.last_log_time = now
    
    def _log_progress(self, now: float) -> None:
        """
        Log the current progress.
        
        Args:
            now: Current timestamp
        """
        elapsed = now - self.start_time
        percent = 100.0 * self.completed / self.total
        
        # Calculate items per second and ETA
        if elapsed > 0 and self.completed > 0:
            items_per_sec = self.completed / elapsed
            remaining = (self.total - self.completed) / items_per_sec if items_per_sec > 0 else 0
            
            # Format ETA as m:ss or h:mm:ss
            if remaining < 3600:
                eta = f"{int(remaining // 60)}:{int(remaining % 60):02d}"
            else:
                eta = f"{int(remaining // 3600)}:{int((remaining % 3600) // 60):02d}:{int(remaining % 60):02d}"
                
            progress_msg = (
                f"{self.desc}: {self.completed}/{self.total} "
                f"({percent:.1f}%) - {items_per_sec:.1f} items/s - ETA: {eta}"
            )
        else:
            progress_msg = f"{self.desc}: {self.completed}/{self.total} ({percent:.1f}%)"
        
        logger.info(progress_msg)
    
    def complete(self) -> None:
        """
        Mark progress as complete and log final status.
        """
        with self._lock:
            self.completed = self.total
            elapsed = time.time() - self.start_time
            
            # Format elapsed time
            if elapsed < 60:
                elapsed_str = f"{elapsed:.1f}s"
            elif elapsed < 3600:
                elapsed_str = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"
            else:
                elapsed_str = f"{int(elapsed // 3600)}h {int((elapsed % 3600) // 60)}m {int(elapsed % 60)}s"
            
            logger.info(f"{self.desc}: Completed {self.total} items in {elapsed_str}")


class ParallelProcessor:
    """
    Enhanced parallel processing with resource management and error propagation.
    """
    
    def __init__(self, max_workers: Optional[int] = None):
        """
        Initialize the parallel processor.
        
        Args:
            max_workers: Maximum number of worker threads/processes (None for auto)
        """
        self.max_workers = max_workers
        self._shutdown_event = threading.Event()
    
    def get_optimal_workers(self, num_tasks: int, use_threads: bool) -> int:
        """
        Determine the optimal number of workers based on system load and resources.
        
        Args:
            num_tasks: Number of tasks to process
            use_threads: Whether using thread pool (vs process pool)
            
        Returns:
            Optimal number of workers
        """
        if self.max_workers is not None:
            return min(self.max_workers, num_tasks)
        
        try:
            # Get system resource information
            cpu_count = os.cpu_count() or multiprocessing.cpu_count()
            
            if use_threads:
                # For threads, consider CPU utilization
                try:
                    cpu_usage = psutil.cpu_percent(interval=0.1) / 100
                    
                    # Adjust worker count based on current CPU usage
                    # More workers for threads since they're lightweight
                    if cpu_usage < 0.3:
                        # Low usage, can use more threads
                        worker_count = int(cpu_count * 4)
                    elif cpu_usage < 0.7:
                        # Medium usage
                        worker_count = int(cpu_count * 2)
                    else:
                        # High usage
                        worker_count = cpu_count
                        
                    return min(max(worker_count, 2), num_tasks)
                except (ImportError, AttributeError):
                    # If psutil not available, use CPU count * 2
                    return min(cpu_count * 2, num_tasks)
            else:
                # For processes, consider memory too
                try:
                    mem = psutil.virtual_memory()
                    mem_available_gb = mem.available / (1024**3)
                    
                    # Estimate 200MB per process (adjust based on your workload)
                    max_by_mem = int(mem_available_gb * 5)  # 5 processes per GB
                    
                    # Take the minimum of CPU count and memory constraint
                    worker_count = min(cpu_count, max_by_mem)
                    return min(max(worker_count, 1), num_tasks)
                except (ImportError, AttributeError):
                    # If psutil not available, use CPU count
                    return min(cpu_count, num_tasks)
        except Exception as e:
            logger.warning(f"Error determining optimal worker count: {e}")
            # Fallback to a reasonable default
            return min(4 if use_threads else 2, num_tasks)
    
    def shutdown(self) -> None:
        """Request shutdown of any running tasks."""
        self._shutdown_event.set()
    
    def is_shutdown_requested(self) -> bool:
        """Check if shutdown has been requested."""
        return self._shutdown_event.is_set()
    
    def process(
        self, 
        tasks: List[Tuple[Callable[..., T], Dict[str, Any]]], 
        use_threads: bool = True,
        description: str = "Processing tasks",
        raise_on_error: bool = False,
        log_interval: float = 1.0,
        timeout: Optional[float] = None
    ) -> List[TaskResult[T]]:
        """
        Process tasks in parallel with enhanced error handling.
        
        Args:
            tasks: List of (function, kwargs) tuples to execute
            use_threads: Whether to use thread pool (True) or process pool (False)
            description: Description for logging
            raise_on_error: Whether to raise the first error encountered
            log_interval: Minimum interval between progress logs in seconds
            timeout: Optional timeout for the entire processing in seconds
            
        Returns:
            List of TaskResult objects in the same order as tasks
            
        Raises:
            Exception: First error encountered if raise_on_error is True
        """
        if not tasks:
            logger.warning("No tasks provided to ParallelProcessor")
            return []
            
        # Reset shutdown event
        self._shutdown_event.clear()
        
        num_tasks = len(tasks)
        executor_cls = concurrent.futures.ThreadPoolExecutor if use_threads else concurrent.futures.ProcessPoolExecutor
        
        # Determine optimal worker count
        workers = self.get_optimal_workers(num_tasks, use_threads)
        
        logger.info(f"Starting {num_tasks} parallel tasks using {workers} workers ({description})")
        start_time = time.time()
        
        # Create progress tracker
        progress = ProgressTracker(num_tasks, description, log_interval)
        
        # Results container (initialized with None for proper ordering)
        results = [TaskResult(task_id=i) for i in range(num_tasks)]
        first_error = None
        
        # Worker function that wraps the actual task function
        def worker_func(func, kwargs, task_id):
            if self.is_shutdown_requested():
                return TaskResult(error=concurrent.futures.CancelledError(), task_id=task_id)
            
            start = time.time()
            try:
                result = func(**kwargs)
                return TaskResult(result=result, task_id=task_id, duration=time.time() - start)
            except Exception as e:
                logger.error(f"Task {task_id+1}/{num_tasks} failed: {e}")
                return TaskResult(error=e, task_id=task_id, duration=time.time() - start)
        
        # Execute tasks
        try:
            with executor_cls(max_workers=workers) as executor:
                # Submit all tasks
                future_to_idx = {}
                for i, (func, kwargs) in enumerate(tasks):
                    future = executor.submit(worker_func, func, kwargs, i)
                    future_to_idx[future] = i
                
                # Track completed tasks and handle timeouts
                completed = 0
                remaining_timeout = timeout
                
                # Process results as they complete
                for future in concurrent.futures.as_completed(future_to_idx, timeout=remaining_timeout):
                    # Update timeout if needed
                    if timeout is not None:
                        elapsed = time.time() - start_time
                        remaining_timeout = max(0.001, timeout - elapsed)
                    
                    # Get task result
                    idx = future_to_idx[future]
                    try:
                        task_result = future.result(timeout=remaining_timeout)
                        results[idx] = task_result
                        
                        # Store first error for potential re-raising
                        if task_result.error and first_error is None:
                            first_error = task_result.error
                            
                            # Early termination if requested
                            if raise_on_error:
                                self.shutdown()
                                break
                    except concurrent.futures.TimeoutError:
                        # Handle timeout for individual result retrieval
                        results[idx] = TaskResult(
                            error=concurrent.futures.TimeoutError(
                                f"Timed out retrieving result for task {idx}"
                            ),
                            task_id=idx
                        )
                    except Exception as e:
                        # Handle unexpected errors
                        results[idx] = TaskResult(error=e, task_id=idx)
                        if first_error is None:
                            first_error = e
                    
                    # Update progress
                    completed += 1
                    progress.increment()
        except concurrent.futures.TimeoutError:
            # Handle timeout for the entire process
            logger.error(f"Parallel processing timed out after {timeout} seconds")
            # Mark all pending tasks as timed out
            for i, result in enumerate(results):
                if not result.success and result.error is None:
                    results[i] = TaskResult(
                        error=concurrent.futures.TimeoutError("Processing timed out"),
                        task_id=i
                    )
        finally:
            # Mark progress as complete
            progress.complete()
        
        # Log overall results
        successful = sum(1 for r in results if r.success)
        logger.info(f"{description}: {successful}/{num_tasks} tasks completed successfully in {time.time() - start_time:.2f}s")
        
        # Raise first error if requested
        if raise_on_error and first_error is not None:
            raise first_error
        
        return results
    
    def process_items(
        self,
        items: List[T],
        process_func: Callable[[T, Dict[str, Any]], R],
        use_threads: bool = True,
        func_kwargs: Optional[Dict[str, Any]] = None,
        description: str = "Processing items",
        raise_on_error: bool = False,
        log_interval: float = 1.0,
        timeout: Optional[float] = None
    ) -> List[TaskResult[R]]:
        """
        Process a list of items in parallel.
        
        Args:
            items: List of items to process
            process_func: Function to apply to each item, with signature func(item, **kwargs)
            use_threads: Whether to use thread pool
            func_kwargs: Additional keyword arguments to pass to the processing function
            description: Description for logging
            raise_on_error: Whether to raise the first error encountered
            log_interval: Minimum interval between progress logs in seconds
            timeout: Optional timeout for the entire processing in seconds
            
        Returns:
            List of TaskResult objects in the same order as items
        """
        if not items:
            logger.warning("No items provided to process_items")
            return []
        
        # Build task list
        tasks = []
        for item in items:
            kwargs = {"item": item}
            if func_kwargs:
                kwargs.update(func_kwargs)
            tasks.append((process_func, kwargs))
        
        # Process tasks
        return self.process(
            tasks,
            use_threads=use_threads,
            description=description,
            raise_on_error=raise_on_error,
            log_interval=log_interval,
            timeout=timeout
        )
    
    def process_batched(
        self,
        items: List[T],
        process_func: Callable[[List[T], Dict[str, Any]], List[R]],
        batch_size: int = 10,
        use_threads: bool = True,
        func_kwargs: Optional[Dict[str, Any]] = None,
        description: str = "Processing batched items",
        raise_on_error: bool = False,
        timeout: Optional[float] = None
    ) -> List[R]:
        """
        Process items in batches for better efficiency.
        
        Args:
            items: List of items to process
            process_func: Function to process each batch, with signature: func(batch_items, **kwargs)
            batch_size: Number of items in each batch
            use_threads: Whether to use thread pool
            func_kwargs: Additional keyword arguments for the processing function
            description: Description for logging
            raise_on_error: Whether to raise the first error encountered
            timeout: Optional timeout for the entire processing in seconds
            
        Returns:
            Flattened list of results from all batches
        """
        if not items:
            logger.warning("No items provided to process_batched")
            return []
        
        # Calculate optimal batch size based on item count and system resources
        if batch_size <= 0:
            batch_size = max(1, min(100, len(items) // self.get_optimal_workers(len(items), use_threads)))
        
        # Create batches
        batches = [items[i:i+batch_size] for i in range(0, len(items), batch_size)]
        num_batches = len(batches)
        
        logger.info(f"Processing {len(items)} items in {num_batches} batches (batch size: {batch_size})")
        
        # Create tasks for each batch
        tasks = []
        for batch in batches:
            kwargs = {"batch_items": batch}
            if func_kwargs:
                kwargs.update(func_kwargs)
            tasks.append((process_func, kwargs))
        
        # Process batches
        batch_results = self.process(
            tasks,
            use_threads=use_threads,
            description=description,
            raise_on_error=raise_on_error,
            timeout=timeout
        )
        
        # Flatten results
        results = []
        for task_result in batch_results:
            if task_result.success and task_result.result:
                results.extend(task_result.result)
        
        logger.info(f"Processed {len(results)} results from {num_batches} batches")
        return results

    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with proper cleanup."""
        self.shutdown()
        return False  # Don't suppress exceptions