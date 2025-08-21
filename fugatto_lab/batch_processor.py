"""Batch Processing Engine for Fugatto Audio Lab.

Efficient batch processing for large-scale audio generation and transformation tasks.
Includes parallel processing, progress tracking, and resource management.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from pathlib import Path
import time
import json
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import threading
from queue import Queue, Empty

# Conditional imports
try:
    from .simple_api import SimpleAudioAPI
    HAS_SIMPLE_API = True
except ImportError:
    HAS_SIMPLE_API = False

try:
    from .quantum_planner import QuantumTaskPlanner, QuantumTask, TaskPriority
    HAS_QUANTUM_PLANNER = True
except ImportError:
    HAS_QUANTUM_PLANNER = False

logger = logging.getLogger(__name__)


@dataclass
class BatchTask:
    """Represents a single task in a batch operation."""
    id: str
    task_type: str  # 'generate' or 'transform'
    parameters: Dict[str, Any]
    priority: int = 0
    status: str = 'pending'
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class BatchProgress:
    """Tracks progress of batch operations."""
    total_tasks: int
    completed_tasks: int = 0
    failed_tasks: int = 0
    pending_tasks: int = 0
    processing_tasks: int = 0
    start_time: float = field(default_factory=time.time)
    estimated_completion: Optional[float] = None
    
    @property
    def completion_rate(self) -> float:
        """Get completion rate as percentage."""
        return (self.completed_tasks / self.total_tasks * 100) if self.total_tasks > 0 else 0
    
    @property
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        return time.time() - self.start_time
    
    @property
    def average_task_time(self) -> float:
        """Get average time per completed task."""
        return self.elapsed_time / self.completed_tasks if self.completed_tasks > 0 else 0


class BatchProcessor:
    """High-performance batch processor for audio operations."""
    
    def __init__(self, max_workers: int = 4, enable_gpu_batching: bool = True,
                 progress_callback: Optional[Callable[[BatchProgress], None]] = None):
        """Initialize batch processor.
        
        Args:
            max_workers: Maximum number of parallel workers
            enable_gpu_batching: Whether to use GPU acceleration for batching
            progress_callback: Optional callback for progress updates
        """
        self.max_workers = max_workers
        self.enable_gpu_batching = enable_gpu_batching
        self.progress_callback = progress_callback
        
        # Task management
        self.task_queue = Queue()
        self.results_queue = Queue()
        self.active_tasks: Dict[str, BatchTask] = {}
        self.completed_tasks: Dict[str, BatchTask] = {}
        
        # Progress tracking
        self.progress = BatchProgress(total_tasks=0)
        self.progress_lock = threading.Lock()
        
        # Processing state
        self.is_processing = False
        self.stop_requested = False
        
        # Audio API instances (one per worker to avoid conflicts)
        self._api_pool: List[SimpleAudioAPI] = []
        
        # Quantum planner for intelligent scheduling
        self._planner = None
        if HAS_QUANTUM_PLANNER:
            self._planner = QuantumTaskPlanner()
        
        logger.info(f"BatchProcessor initialized with {max_workers} workers")
    
    def add_generation_task(self, task_id: str, prompt: str, duration: float = 10.0,
                           output_path: Optional[str] = None, priority: int = 0,
                           **kwargs) -> str:
        """Add audio generation task to batch.
        
        Args:
            task_id: Unique identifier for the task
            prompt: Audio description
            duration: Audio length in seconds
            output_path: Output file path
            priority: Task priority (higher = more important)
            **kwargs: Additional generation parameters
            
        Returns:
            Task ID
        """
        task = BatchTask(
            id=task_id,
            task_type='generate',
            parameters={
                'prompt': prompt,
                'duration': duration,
                'output_path': output_path,
                **kwargs
            },
            priority=priority
        )
        
        self.task_queue.put(task)
        logger.debug(f"Added generation task: {task_id}")
        return task_id
    
    def add_transformation_task(self, task_id: str, input_path: str, prompt: str,
                               output_path: Optional[str] = None, strength: float = 0.7,
                               priority: int = 0, **kwargs) -> str:
        """Add audio transformation task to batch.
        
        Args:
            task_id: Unique identifier for the task
            input_path: Input audio file path
            prompt: Transformation description
            output_path: Output file path
            strength: Transformation strength
            priority: Task priority
            **kwargs: Additional transformation parameters
            
        Returns:
            Task ID
        """
        task = BatchTask(
            id=task_id,
            task_type='transform',
            parameters={
                'input_path': input_path,
                'prompt': prompt,
                'output_path': output_path,
                'strength': strength,
                **kwargs
            },
            priority=priority
        )
        
        self.task_queue.put(task)
        logger.debug(f"Added transformation task: {task_id}")
        return task_id
    
    def add_tasks_from_config(self, config_path: Union[str, Path]) -> List[str]:
        """Load tasks from JSON configuration file.
        
        Args:
            config_path: Path to JSON configuration file
            
        Returns:
            List of task IDs that were added
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        task_ids = []
        
        # Process generation tasks
        for i, task_config in enumerate(config.get('generation_tasks', [])):
            task_id = task_config.get('id', f"gen_{i}")
            self.add_generation_task(task_id, **task_config)
            task_ids.append(task_id)
        
        # Process transformation tasks
        for i, task_config in enumerate(config.get('transformation_tasks', [])):
            task_id = task_config.get('id', f"transform_{i}")
            self.add_transformation_task(task_id, **task_config)
            task_ids.append(task_id)
        
        logger.info(f"Loaded {len(task_ids)} tasks from {config_path}")
        return task_ids
    
    def start_processing(self) -> None:
        """Start batch processing in background."""
        if self.is_processing:
            logger.warning("Batch processing already in progress")
            return
        
        self.is_processing = True
        self.stop_requested = False
        
        # Count tasks for progress tracking
        with self.progress_lock:
            self.progress.total_tasks = self.task_queue.qsize()
            self.progress.pending_tasks = self.progress.total_tasks
        
        logger.info(f"Starting batch processing with {self.progress.total_tasks} tasks")
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._process_batch, daemon=True)
        self.processing_thread.start()
    
    def stop_processing(self, wait: bool = True) -> None:
        """Stop batch processing.
        
        Args:
            wait: Whether to wait for current tasks to complete
        """
        logger.info("Stopping batch processing...")
        self.stop_requested = True
        
        if wait and hasattr(self, 'processing_thread'):
            self.processing_thread.join(timeout=30)
        
        self.is_processing = False
        logger.info("Batch processing stopped")
    
    def _process_batch(self) -> None:
        """Main batch processing loop."""
        # Initialize API pool
        self._initialize_api_pool()
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            
            while not self.stop_requested:
                # Submit new tasks up to worker limit
                while len(futures) < self.max_workers and not self.task_queue.empty():
                    try:
                        task = self.task_queue.get_nowait()
                        
                        # Schedule with quantum planner if available
                        if self._planner:
                            try:
                                from .quantum_planner import TaskPriority
                                quantum_task = QuantumTask(
                                    id=task.id,
                                    name=f"{task.task_type}_{task.id}",
                                    description=f"Batch {task.task_type} task",
                                    priority=TaskPriority.HIGH if task.priority > 5 else TaskPriority.MEDIUM,
                                    estimated_duration=30.0,
                                    context={'operation': task.task_type, **task.parameters}
                                )
                                self._planner.add_task(quantum_task)
                                optimized_params = self._planner.optimize_task_execution(quantum_task.id)
                                if optimized_params:
                                    task.parameters.update(optimized_params)
                            except Exception as e:
                                logger.debug(f"Quantum planning failed for task {task.id}: {e}")
                        
                        # Submit task
                        task.status = 'processing'
                        task.started_at = time.time()
                        self.active_tasks[task.id] = task
                        
                        future = executor.submit(self._process_single_task, task)
                        futures[future] = task
                        
                        self._update_progress()
                        
                    except Empty:
                        break
                
                # Check completed tasks
                for future in as_completed(futures, timeout=0.1):
                    task = futures.pop(future)
                    
                    try:
                        result = future.result()
                        task.result = result
                        task.status = 'completed'
                        task.completed_at = time.time()
                        
                        logger.debug(f"Task {task.id} completed successfully")
                        
                    except Exception as e:
                        task.error = str(e)
                        task.status = 'failed'
                        task.completed_at = time.time()
                        
                        # Retry logic
                        if task.retry_count < task.max_retries:
                            task.retry_count += 1
                            task.status = 'pending'
                            task.error = None
                            self.task_queue.put(task)
                            logger.warning(f"Task {task.id} failed, retrying ({task.retry_count}/{task.max_retries})")
                        else:
                            logger.error(f"Task {task.id} failed permanently: {e}")
                    
                    # Move to completed tasks
                    self.completed_tasks[task.id] = task
                    del self.active_tasks[task.id]
                    
                    self._update_progress()
                    break  # Process one completed task per iteration
                
                # Check if all tasks are done
                if self.task_queue.empty() and not futures:
                    break
                
                # Brief pause to prevent CPU spinning
                time.sleep(0.01)
        
        self.is_processing = False
        logger.info("Batch processing completed")
    
    def _process_single_task(self, task: BatchTask) -> Dict[str, Any]:
        """Process a single task.
        
        Args:
            task: Task to process
            
        Returns:
            Task result dictionary
        """
        # Get API instance from pool
        api = self._get_api_instance()
        
        try:
            if task.task_type == 'generate':
                result = api.generate_audio(**task.parameters)
            elif task.task_type == 'transform':
                result = api.transform_audio(**task.parameters)
            else:
                raise ValueError(f"Unknown task type: {task.task_type}")
            
            return result
            
        finally:
            # Return API instance to pool
            self._return_api_instance(api)
    
    def _initialize_api_pool(self) -> None:
        """Initialize pool of API instances for parallel processing."""
        if not HAS_SIMPLE_API:
            logger.warning("SimpleAudioAPI not available, using mock processing")
            return
        
        self._api_pool = []
        for i in range(self.max_workers):
            try:
                api = SimpleAudioAPI()
                self._api_pool.append(api)
                logger.debug(f"Initialized API instance {i+1}/{self.max_workers}")
            except Exception as e:
                logger.error(f"Failed to initialize API instance {i+1}: {e}")
    
    def _get_api_instance(self) -> Optional[SimpleAudioAPI]:
        """Get an API instance from the pool."""
        if self._api_pool:
            return self._api_pool.pop()
        elif HAS_SIMPLE_API:
            # Create new instance if pool is empty
            return SimpleAudioAPI()
        else:
            return None
    
    def _return_api_instance(self, api: Optional[SimpleAudioAPI]) -> None:
        """Return an API instance to the pool."""
        if api and len(self._api_pool) < self.max_workers:
            self._api_pool.append(api)
    
    def _update_progress(self) -> None:
        """Update progress tracking and call callback if provided."""
        with self.progress_lock:
            # Update counts
            self.progress.pending_tasks = self.task_queue.qsize()
            self.progress.processing_tasks = len(self.active_tasks)
            self.progress.completed_tasks = len([t for t in self.completed_tasks.values() if t.status == 'completed'])
            self.progress.failed_tasks = len([t for t in self.completed_tasks.values() if t.status == 'failed'])
            
            # Estimate completion time
            if self.progress.completed_tasks > 0:
                avg_time = self.progress.average_task_time
                remaining_tasks = self.progress.total_tasks - self.progress.completed_tasks
                self.progress.estimated_completion = time.time() + (avg_time * remaining_tasks)
        
        # Call progress callback
        if self.progress_callback:
            try:
                self.progress_callback(self.progress)
            except Exception as e:
                logger.error(f"Progress callback failed: {e}")
    
    def get_progress(self) -> BatchProgress:
        """Get current batch processing progress.
        
        Returns:
            Current progress information
        """
        with self.progress_lock:
            return BatchProgress(
                total_tasks=self.progress.total_tasks,
                completed_tasks=self.progress.completed_tasks,
                failed_tasks=self.progress.failed_tasks,
                pending_tasks=self.progress.pending_tasks,
                processing_tasks=self.progress.processing_tasks,
                start_time=self.progress.start_time,
                estimated_completion=self.progress.estimated_completion
            )
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Task status information or None if not found
        """
        # Check active tasks
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
        elif task_id in self.completed_tasks:
            task = self.completed_tasks[task_id]
        else:
            return None
        
        return {
            'id': task.id,
            'type': task.task_type,
            'status': task.status,
            'priority': task.priority,
            'created_at': task.created_at,
            'started_at': task.started_at,
            'completed_at': task.completed_at,
            'retry_count': task.retry_count,
            'processing_time': (task.completed_at - task.started_at) if task.started_at and task.completed_at else None,
            'error': task.error,
            'result_available': task.result is not None
        }
    
    def get_task_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get result of a completed task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Task result or None if not available
        """
        if task_id in self.completed_tasks:
            return self.completed_tasks[task_id].result
        return None
    
    def export_results(self, output_path: Union[str, Path]) -> None:
        """Export batch processing results to JSON file.
        
        Args:
            output_path: Path to output JSON file
        """
        output_path = Path(output_path)
        
        # Collect all results
        results = {
            'batch_summary': {
                'total_tasks': self.progress.total_tasks,
                'completed_tasks': self.progress.completed_tasks,
                'failed_tasks': self.progress.failed_tasks,
                'processing_time': self.progress.elapsed_time,
                'average_task_time': self.progress.average_task_time,
                'completion_rate': self.progress.completion_rate
            },
            'tasks': {}
        }
        
        # Add task details
        for task_id, task in self.completed_tasks.items():
            results['tasks'][task_id] = {
                'id': task.id,
                'type': task.task_type,
                'status': task.status,
                'parameters': task.parameters,
                'created_at': task.created_at,
                'started_at': task.started_at,
                'completed_at': task.completed_at,
                'processing_time': (task.completed_at - task.started_at) if task.started_at and task.completed_at else None,
                'retry_count': task.retry_count,
                'error': task.error,
                'result': task.result
            }
        
        # Save to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Batch results exported to {output_path}")
    
    def clear_completed_tasks(self) -> int:
        """Clear completed tasks to free memory.
        
        Returns:
            Number of tasks cleared
        """
        count = len(self.completed_tasks)
        self.completed_tasks.clear()
        logger.info(f"Cleared {count} completed tasks")
        return count


# Convenience functions
def process_batch_config(config_path: Union[str, Path], max_workers: int = 4,
                        output_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Process a batch configuration file and return results.
    
    Args:
        config_path: Path to batch configuration JSON file
        max_workers: Number of parallel workers
        output_path: Path to save results (optional)
        
    Returns:
        Batch processing results
    """
    processor = BatchProcessor(max_workers=max_workers)
    
    # Load tasks
    task_ids = processor.add_tasks_from_config(config_path)
    
    # Process batch
    processor.start_processing()
    
    # Wait for completion (with progress updates)
    while processor.is_processing:
        progress = processor.get_progress()
        logger.info(f"Progress: {progress.completion_rate:.1f}% ({progress.completed_tasks}/{progress.total_tasks})")
        time.sleep(5)
    
    # Get final results
    final_progress = processor.get_progress()
    
    # Export results if requested
    if output_path:
        processor.export_results(output_path)
    
    return {
        'task_ids': task_ids,
        'progress': final_progress,
        'results_exported': output_path is not None,
        'output_path': str(output_path) if output_path else None
    }


if __name__ == "__main__":
    # Demo batch processing
    import tempfile
    
    # Create sample configuration
    config = {
        "generation_tasks": [
            {
                "id": "rain_sample",
                "prompt": "Gentle rain on leaves",
                "duration": 5.0,
                "priority": 2
            },
            {
                "id": "bird_sample", 
                "prompt": "Bird singing in the morning",
                "duration": 3.0,
                "priority": 1
            },
            {
                "id": "ocean_sample",
                "prompt": "Ocean waves on a beach",
                "duration": 4.0,
                "priority": 3
            }
        ]
    }
    
    # Save config to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config, f, indent=2)
        config_path = f.name
    
    # Process batch
    logger.info("Starting batch processing demo...")
    results = process_batch_config(config_path, max_workers=2)
    logger.info(f"Demo completed: {results}")
    
    # Clean up
    Path(config_path).unlink()