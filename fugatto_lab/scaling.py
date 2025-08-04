"""Scaling and concurrency utilities for Fugatto Audio Lab."""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Dict, Any, List, Optional, Callable, Awaitable
import time
import multiprocessing as mp
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ScalingConfig:
    """Configuration for scaling operations."""
    max_workers: int = mp.cpu_count()
    use_process_pool: bool = False
    enable_gpu_parallel: bool = False
    batch_size: int = 4
    timeout_seconds: float = 60.0


class ConcurrentAudioProcessor:
    """Concurrent audio processing with thread/process pools."""
    
    def __init__(self, config: Optional[ScalingConfig] = None):
        """Initialize concurrent processor.
        
        Args:
            config: Scaling configuration
        """
        self.config = config or ScalingConfig()
        
        if self.config.use_process_pool:
            self.executor = ProcessPoolExecutor(max_workers=self.config.max_workers)
            logger.info(f"Initialized ProcessPoolExecutor with {self.config.max_workers} workers")
        else:
            self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
            logger.info(f"Initialized ThreadPoolExecutor with {self.config.max_workers} workers")
    
    async def process_batch_async(self, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process batch of requests asynchronously.
        
        Args:
            requests: List of processing requests
            
        Returns:
            List of processing results
        """
        loop = asyncio.get_event_loop()
        
        # Create tasks for concurrent processing
        tasks = []
        for request in requests:
            task = loop.run_in_executor(
                self.executor, 
                self._process_single_request, 
                request
            )
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Request {i} failed: {result}")
                processed_results.append({
                    'success': False,
                    'error': str(result),
                    'request_id': requests[i].get('id', f'req_{i}')
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    def _process_single_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process single request (to be run in executor).
        
        Args:
            request: Processing request
            
        Returns:
            Processing result
        """
        start_time = time.time()
        
        try:
            # Simulate audio processing
            prompt = request.get('prompt', 'test')
            duration = request.get('duration_seconds', 5.0)
            
            # Mock processing time
            time.sleep(0.1)  # Simulate processing
            
            # Generate mock audio
            sample_rate = 48000
            num_samples = int(duration * sample_rate)
            audio = np.random.randn(num_samples).astype(np.float32) * 0.1
            
            processing_time = (time.time() - start_time) * 1000
            
            return {
                'success': True,
                'request_id': request.get('id', 'unknown'),
                'prompt': prompt,
                'duration_seconds': duration,
                'processing_time_ms': processing_time,
                'audio_samples': len(audio),
                'worker_pid': mp.current_process().pid
            }
            
        except Exception as e:
            logger.error(f"Processing error: {e}")
            return {
                'success': False,
                'request_id': request.get('id', 'unknown'),
                'error': str(e),
                'processing_time_ms': (time.time() - start_time) * 1000
            }
    
    def shutdown(self):
        """Shutdown the executor."""
        self.executor.shutdown(wait=True)
        logger.info("Concurrent processor shutdown complete")


class LoadBalancer:
    """Simple load balancer for distributing requests."""
    
    def __init__(self, workers: List[str]):
        """Initialize load balancer.
        
        Args:
            workers: List of worker identifiers
        """
        self.workers = workers
        self.current_worker = 0
        self.request_counts = {worker: 0 for worker in workers}
        
        logger.info(f"LoadBalancer initialized with {len(workers)} workers")
    
    def get_next_worker(self) -> str:
        """Get next worker using round-robin."""
        worker = self.workers[self.current_worker]
        self.current_worker = (self.current_worker + 1) % len(self.workers)
        self.request_counts[worker] += 1
        return worker
    
    def get_least_loaded_worker(self) -> str:
        """Get worker with least requests."""
        worker = min(self.request_counts.keys(), key=lambda w: self.request_counts[w])
        self.request_counts[worker] += 1
        return worker
    
    def get_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        total_requests = sum(self.request_counts.values())
        return {
            'total_workers': len(self.workers),
            'total_requests': total_requests,
            'requests_per_worker': dict(self.request_counts),
            'avg_requests_per_worker': total_requests / len(self.workers) if self.workers else 0
        }


class AutoScaler:
    """Automatic scaling based on load."""
    
    def __init__(self, min_workers: int = 1, max_workers: int = 8, target_cpu_percent: float = 70.0):
        """Initialize auto-scaler.
        
        Args:
            min_workers: Minimum number of workers
            max_workers: Maximum number of workers
            target_cpu_percent: Target CPU utilization percentage
        """
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.target_cpu_percent = target_cpu_percent
        self.current_workers = min_workers
        
        logger.info(f"AutoScaler initialized: {min_workers}-{max_workers} workers, target CPU {target_cpu_percent}%")
    
    def should_scale_up(self, current_cpu_percent: float, queue_length: int) -> bool:
        """Determine if should scale up.
        
        Args:
            current_cpu_percent: Current CPU utilization
            queue_length: Current queue length
            
        Returns:
            True if should scale up
        """
        return (
            self.current_workers < self.max_workers and 
            (current_cpu_percent > self.target_cpu_percent or queue_length > self.current_workers * 2)
        )
    
    def should_scale_down(self, current_cpu_percent: float, queue_length: int) -> bool:
        """Determine if should scale down.
        
        Args:
            current_cpu_percent: Current CPU utilization
            queue_length: Current queue length
            
        Returns:
            True if should scale down
        """
        return (
            self.current_workers > self.min_workers and 
            current_cpu_percent < self.target_cpu_percent * 0.5 and 
            queue_length < self.current_workers
        )
    
    def get_scaling_recommendation(self, current_cpu_percent: float, queue_length: int) -> Dict[str, Any]:
        """Get scaling recommendation.
        
        Args:
            current_cpu_percent: Current CPU utilization
            queue_length: Current queue length
            
        Returns:
            Scaling recommendation
        """
        recommendation = {
            'action': 'maintain',
            'current_workers': self.current_workers,
            'recommended_workers': self.current_workers,
            'reason': 'Current capacity is adequate'
        }
        
        if self.should_scale_up(current_cpu_percent, queue_length):
            new_workers = min(self.current_workers + 1, self.max_workers)
            recommendation.update({
                'action': 'scale_up',
                'recommended_workers': new_workers,
                'reason': f'High load detected: CPU {current_cpu_percent}%, queue {queue_length}'
            })
        elif self.should_scale_down(current_cpu_percent, queue_length):
            new_workers = max(self.current_workers - 1, self.min_workers)
            recommendation.update({
                'action': 'scale_down',
                'recommended_workers': new_workers,
                'reason': f'Low load detected: CPU {current_cpu_percent}%, queue {queue_length}'
            })
        
        return recommendation
    
    def apply_scaling(self, new_worker_count: int):
        """Apply scaling decision.
        
        Args:
            new_worker_count: New number of workers
        """
        old_count = self.current_workers
        self.current_workers = max(self.min_workers, min(new_worker_count, self.max_workers))
        
        if self.current_workers != old_count:
            logger.info(f"Scaled from {old_count} to {self.current_workers} workers")


class DistributedCoordinator:
    """Coordinator for distributed processing."""
    
    def __init__(self):
        """Initialize distributed coordinator."""
        self.nodes = {}
        self.task_queue = asyncio.Queue()
        self.result_store = {}
        
        logger.info("DistributedCoordinator initialized")
    
    async def register_node(self, node_id: str, capabilities: Dict[str, Any]):
        """Register a processing node.
        
        Args:
            node_id: Unique node identifier
            capabilities: Node capabilities and resources
        """
        self.nodes[node_id] = {
            'capabilities': capabilities,
            'last_seen': time.time(),
            'status': 'active',
            'tasks_processed': 0
        }
        
        logger.info(f"Registered node {node_id} with capabilities: {capabilities}")
    
    async def submit_task(self, task: Dict[str, Any]) -> str:
        """Submit task for distributed processing.
        
        Args:
            task: Task to process
            
        Returns:
            Task ID
        """
        task_id = f"task_{int(time.time() * 1000000)}"
        task['id'] = task_id
        task['submitted_at'] = time.time()
        
        await self.task_queue.put(task)
        logger.debug(f"Submitted task {task_id}")
        
        return task_id
    
    async def get_result(self, task_id: str, timeout: float = 30.0) -> Optional[Dict[str, Any]]:
        """Get task result.
        
        Args:
            task_id: Task identifier
            timeout: Timeout in seconds
            
        Returns:
            Task result or None if timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if task_id in self.result_store:
                result = self.result_store.pop(task_id)
                logger.debug(f"Retrieved result for task {task_id}")
                return result
            
            await asyncio.sleep(0.1)  # Poll every 100ms
        
        logger.warning(f"Timeout waiting for result of task {task_id}")
        return None
    
    def get_cluster_stats(self) -> Dict[str, Any]:
        """Get distributed cluster statistics."""
        active_nodes = sum(1 for node in self.nodes.values() if node['status'] == 'active')
        total_tasks = sum(node['tasks_processed'] for node in self.nodes.values())
        
        return {
            'total_nodes': len(self.nodes),
            'active_nodes': active_nodes,
            'total_tasks_processed': total_tasks,
            'queue_size': self.task_queue.qsize(),
            'pending_results': len(self.result_store)
        }


# Global instances
_concurrent_processor = None
_load_balancer = None
_auto_scaler = None
_distributed_coordinator = None


def get_concurrent_processor(config: Optional[ScalingConfig] = None) -> ConcurrentAudioProcessor:
    """Get global concurrent processor instance."""
    global _concurrent_processor
    if _concurrent_processor is None:
        _concurrent_processor = ConcurrentAudioProcessor(config)
    return _concurrent_processor


def get_load_balancer(workers: Optional[List[str]] = None) -> LoadBalancer:
    """Get global load balancer instance."""
    global _load_balancer
    if _load_balancer is None:
        workers = workers or [f"worker_{i}" for i in range(4)]
        _load_balancer = LoadBalancer(workers)
    return _load_balancer


def get_auto_scaler() -> AutoScaler:
    """Get global auto-scaler instance."""
    global _auto_scaler
    if _auto_scaler is None:
        _auto_scaler = AutoScaler()
    return _auto_scaler


def get_distributed_coordinator() -> DistributedCoordinator:
    """Get global distributed coordinator instance."""
    global _distributed_coordinator
    if _distributed_coordinator is None:
        _distributed_coordinator = DistributedCoordinator()
    return _distributed_coordinator