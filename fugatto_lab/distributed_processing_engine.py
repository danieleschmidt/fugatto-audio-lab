"""
Distributed Processing Engine with Auto-Scaling and Load Balancing
Generation 3: Advanced Distributed Computing and Horizontal Scaling
"""

import time
import threading
import multiprocessing
import asyncio
import uuid
import pickle
import json
import hashlib
import logging
from typing import Dict, Any, Optional, List, Union, Callable, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future, as_completed
import queue
import socket
import struct
from pathlib import Path

# Distributed processing components
class NodeType(Enum):
    """Types of nodes in the distributed system."""
    COORDINATOR = "coordinator"
    WORKER = "worker"
    STORAGE = "storage"
    CACHE = "cache"
    LOAD_BALANCER = "load_balancer"
    MONITOR = "monitor"

class TaskPriority(Enum):
    """Task priority levels for scheduling."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
    REALTIME = 5

class ScalingStrategy(Enum):
    """Auto-scaling strategies."""
    CPU_BASED = "cpu_based"
    MEMORY_BASED = "memory_based"
    QUEUE_LENGTH = "queue_length"
    RESPONSE_TIME = "response_time"
    PREDICTIVE = "predictive"
    HYBRID = "hybrid"

@dataclass
class DistributedTask:
    """Distributed task with serialization support."""
    task_id: str
    function_name: str
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    timeout: float = 300.0  # 5 minutes default
    retry_count: int = 0
    max_retries: int = 3
    dependencies: Set[str] = field(default_factory=set)
    
    # Execution tracking
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    assigned_node: Optional[str] = None
    
    # Result storage
    result: Any = None
    error: Optional[str] = None
    
    # Resource requirements
    cpu_requirement: float = 1.0  # CPU cores
    memory_requirement: int = 512  # MB
    gpu_requirement: bool = False
    
    def __post_init__(self):
        if not self.task_id:
            self.task_id = str(uuid.uuid4())
    
    def serialize(self) -> bytes:
        """Serialize task for network transmission."""
        return pickle.dumps(self)
    
    @classmethod
    def deserialize(cls, data: bytes) -> 'DistributedTask':
        """Deserialize task from network transmission."""
        return pickle.loads(data)

@dataclass
class WorkerNode:
    """Worker node information."""
    node_id: str
    host: str
    port: int
    node_type: NodeType
    
    # Resource capacity
    cpu_cores: int = 4
    memory_mb: int = 8192
    gpu_available: bool = False
    
    # Current utilization
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    active_tasks: int = 0
    
    # Status
    is_healthy: bool = True
    last_heartbeat: float = field(default_factory=time.time)
    
    # Performance metrics
    tasks_completed: int = 0
    tasks_failed: int = 0
    average_task_duration: float = 0.0
    
    def get_load_score(self) -> float:
        """Calculate load score for load balancing."""
        cpu_load = self.cpu_usage / 100.0
        memory_load = self.memory_usage / 100.0
        task_load = self.active_tasks / max(self.cpu_cores, 1)
        
        return (cpu_load + memory_load + task_load) / 3.0
    
    def can_handle_task(self, task: DistributedTask) -> bool:
        """Check if node can handle the task."""
        if task.gpu_requirement and not self.gpu_available:
            return False
        
        if task.cpu_requirement > (self.cpu_cores - self.active_tasks):
            return False
        
        # Estimate memory usage (simplified)
        estimated_memory = self.memory_usage + (task.memory_requirement / self.memory_mb * 100)
        if estimated_memory > 90.0:  # 90% memory threshold
            return False
        
        return True

class DistributedProcessingEngine:
    """
    Advanced distributed processing engine with auto-scaling and load balancing.
    
    Generation 3 Features:
    - Horizontal scaling across multiple nodes
    - Intelligent load balancing algorithms
    - Auto-scaling based on workload
    - Fault-tolerant task distribution
    - Resource-aware scheduling
    - Network optimization and compression
    - Real-time monitoring and metrics
    - Dynamic worker pool management
    - Task dependency resolution
    - Distributed caching system
    """
    
    def __init__(self,
                 node_type: NodeType = NodeType.COORDINATOR,
                 host: str = "localhost",
                 port: int = 8888,
                 enable_auto_scaling: bool = True,
                 max_workers: int = 100,
                 min_workers: int = 2):
        """
        Initialize distributed processing engine.
        
        Args:
            node_type: Type of this node
            host: Host address for this node
            port: Port for this node
            enable_auto_scaling: Enable automatic scaling
            max_workers: Maximum number of worker processes
            min_workers: Minimum number of worker processes
        """
        self.node_type = node_type
        self.host = host
        self.port = port
        self.enable_auto_scaling = enable_auto_scaling
        self.max_workers = max_workers
        self.min_workers = min_workers
        
        # Node identification
        self.node_id = f"{node_type.value}_{host}_{port}_{int(time.time())}"
        
        # Task management
        self.task_queue: queue.PriorityQueue = queue.PriorityQueue()
        self.pending_tasks: Dict[str, DistributedTask] = {}
        self.running_tasks: Dict[str, DistributedTask] = {}
        self.completed_tasks: Dict[str, DistributedTask] = {}
        self.failed_tasks: Dict[str, DistributedTask] = {}
        
        # Worker management
        self.worker_nodes: Dict[str, WorkerNode] = {}
        self.local_workers: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=min_workers)
        self.process_pool: Optional[ProcessPoolExecutor] = None
        
        # Load balancing
        self.load_balancer = LoadBalancer()
        self.task_dispatcher = TaskDispatcher()
        
        # Auto-scaling
        self.auto_scaler = AutoScaler(
            min_workers=min_workers,
            max_workers=max_workers,
            scaling_strategy=ScalingStrategy.HYBRID
        )
        
        # Distributed caching
        self.cache_manager = DistributedCacheManager()
        
        # Networking
        self.network_manager = NetworkManager(host, port)
        self.heartbeat_interval = 30.0  # 30 seconds
        
        # Metrics and monitoring
        self.metrics = {
            'tasks_submitted': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'total_processing_time': 0.0,
            'average_task_duration': 0.0,
            'current_throughput': 0.0,
            'peak_throughput': 0.0,
            'resource_utilization': 0.0,
            'scaling_events': 0,
            'network_bytes_sent': 0,
            'network_bytes_received': 0
        }
        
        # Configuration
        self.config = {
            'task_timeout': 300.0,
            'heartbeat_timeout': 90.0,  # 3 missed heartbeats
            'max_retries': 3,
            'chunk_size': 1024 * 1024,  # 1MB chunks
            'compression_enabled': True,
            'encryption_enabled': True,
            'batch_processing': True,
            'prefetch_tasks': 5,
            'load_balance_algorithm': 'weighted_round_robin'
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Background services
        self._services_running = False
        self._background_threads: List[threading.Thread] = []
        
        # Function registry for distributed execution
        self.function_registry: Dict[str, Callable] = {}
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"DistributedProcessingEngine initialized as {node_type.value} on {host}:{port}")

    def register_function(self, name: str, func: Callable) -> None:
        """
        Register function for distributed execution.
        
        Args:
            name: Function name for remote calls
            func: Function to register
        """
        with self._lock:
            self.function_registry[name] = func
        
        self.logger.info(f"Function '{name}' registered for distributed execution")

    def submit_task(self, function_name: str, *args, 
                   priority: TaskPriority = TaskPriority.NORMAL,
                   timeout: float = 300.0,
                   resource_requirements: Optional[Dict[str, Any]] = None,
                   dependencies: Optional[Set[str]] = None,
                   **kwargs) -> str:
        """
        Submit task for distributed processing.
        
        Args:
            function_name: Name of registered function
            *args: Function arguments
            priority: Task priority
            timeout: Task timeout in seconds
            resource_requirements: Resource requirements
            dependencies: Task dependencies
            **kwargs: Function keyword arguments
            
        Returns:
            Task ID for tracking
        """
        # Create distributed task
        task = DistributedTask(
            task_id="",  # Will be generated
            function_name=function_name,
            args=args,
            kwargs=kwargs,
            priority=priority,
            timeout=timeout,
            dependencies=dependencies or set()
        )
        
        # Set resource requirements
        if resource_requirements:
            task.cpu_requirement = resource_requirements.get('cpu_cores', 1.0)
            task.memory_requirement = resource_requirements.get('memory_mb', 512)
            task.gpu_requirement = resource_requirements.get('gpu', False)
        
        with self._lock:
            self.pending_tasks[task.task_id] = task
            
            # Add to priority queue (lower priority value = higher priority)
            priority_value = priority.value
            self.task_queue.put((priority_value, time.time(), task.task_id))
            
            self.metrics['tasks_submitted'] += 1
        
        # Trigger auto-scaling if needed
        if self.enable_auto_scaling:
            self.auto_scaler.evaluate_scaling_need(self._get_system_metrics())
        
        self.logger.info(f"Task {task.task_id} submitted for function '{function_name}'")
        return task.task_id

    def get_task_result(self, task_id: str, timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Get task result (blocking).
        
        Args:
            task_id: Task ID to get result for
            timeout: Timeout for waiting
            
        Returns:
            Task result dictionary
        """
        start_time = time.time()
        
        while True:
            with self._lock:
                if task_id in self.completed_tasks:
                    task = self.completed_tasks[task_id]
                    return {
                        'success': True,
                        'result': task.result,
                        'task_id': task_id,
                        'execution_time': task.completed_at - task.started_at if task.started_at else 0,
                        'node': task.assigned_node
                    }
                
                if task_id in self.failed_tasks:
                    task = self.failed_tasks[task_id]
                    return {
                        'success': False,
                        'error': task.error,
                        'task_id': task_id,
                        'retry_count': task.retry_count,
                        'node': task.assigned_node
                    }
            
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                return {
                    'success': False,
                    'error': 'timeout',
                    'task_id': task_id
                }
            
            time.sleep(0.1)  # Small delay to avoid busy waiting

    def start_services(self) -> None:
        """Start background services."""
        if self._services_running:
            return
        
        self._services_running = True
        
        # Start task processor
        processor_thread = threading.Thread(
            target=self._task_processor_loop,
            daemon=True,
            name="task_processor"
        )
        processor_thread.start()
        self._background_threads.append(processor_thread)
        
        # Start heartbeat service
        heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,
            name="heartbeat"
        )
        heartbeat_thread.start()
        self._background_threads.append(heartbeat_thread)
        
        # Start metrics collector
        metrics_thread = threading.Thread(
            target=self._metrics_loop,
            daemon=True,
            name="metrics"
        )
        metrics_thread.start()
        self._background_threads.append(metrics_thread)
        
        # Start network service if coordinator
        if self.node_type == NodeType.COORDINATOR:
            network_thread = threading.Thread(
                target=self._network_service_loop,
                daemon=True,
                name="network_service"
            )
            network_thread.start()
            self._background_threads.append(network_thread)
        
        # Initialize process pool for CPU-intensive tasks
        if not self.process_pool:
            self.process_pool = ProcessPoolExecutor(max_workers=multiprocessing.cpu_count())
        
        self.logger.info("Distributed processing services started")

    def stop_services(self) -> None:
        """Stop background services."""
        self._services_running = False
        
        # Shutdown executors
        if self.local_workers:
            self.local_workers.shutdown(wait=True)
        
        if self.process_pool:
            self.process_pool.shutdown(wait=True)
        
        # Wait for background threads to finish
        for thread in self._background_threads:
            if thread.is_alive():
                thread.join(timeout=5.0)
        
        self.logger.info("Distributed processing services stopped")

    def add_worker_node(self, node: WorkerNode) -> None:
        """
        Add worker node to the cluster.
        
        Args:
            node: Worker node to add
        """
        with self._lock:
            self.worker_nodes[node.node_id] = node
        
        self.logger.info(f"Worker node {node.node_id} added to cluster")

    def remove_worker_node(self, node_id: str) -> None:
        """
        Remove worker node from cluster.
        
        Args:
            node_id: Node ID to remove
        """
        with self._lock:
            if node_id in self.worker_nodes:
                del self.worker_nodes[node_id]
                
                # Reschedule tasks from removed node
                self._reschedule_tasks_from_node(node_id)
        
        self.logger.info(f"Worker node {node_id} removed from cluster")

    def get_cluster_status(self) -> Dict[str, Any]:
        """
        Get comprehensive cluster status.
        
        Returns:
            Cluster status information
        """
        with self._lock:
            # Node statistics
            healthy_nodes = len([n for n in self.worker_nodes.values() if n.is_healthy])
            total_cpu_cores = sum(n.cpu_cores for n in self.worker_nodes.values())
            total_memory = sum(n.memory_mb for n in self.worker_nodes.values())
            
            # Task statistics
            task_stats = {
                'pending': len(self.pending_tasks),
                'running': len(self.running_tasks),
                'completed': len(self.completed_tasks),
                'failed': len(self.failed_tasks),
                'queue_length': self.task_queue.qsize()
            }
            
            # Resource utilization
            if self.worker_nodes:
                avg_cpu_usage = sum(n.cpu_usage for n in self.worker_nodes.values()) / len(self.worker_nodes)
                avg_memory_usage = sum(n.memory_usage for n in self.worker_nodes.values()) / len(self.worker_nodes)
            else:
                avg_cpu_usage = 0.0
                avg_memory_usage = 0.0
        
        return {
            'cluster_info': {
                'coordinator': f"{self.host}:{self.port}",
                'total_nodes': len(self.worker_nodes),
                'healthy_nodes': healthy_nodes,
                'total_cpu_cores': total_cpu_cores,
                'total_memory_mb': total_memory
            },
            'task_statistics': task_stats,
            'resource_utilization': {
                'average_cpu_usage': avg_cpu_usage,
                'average_memory_usage': avg_memory_usage,
                'resource_utilization_score': (avg_cpu_usage + avg_memory_usage) / 200.0
            },
            'performance_metrics': self.metrics,
            'auto_scaling': {
                'enabled': self.enable_auto_scaling,
                'current_workers': len(self.worker_nodes),
                'min_workers': self.min_workers,
                'max_workers': self.max_workers,
                'scaling_strategy': self.auto_scaler.scaling_strategy.value
            }
        }

    # Internal processing methods
    
    def _task_processor_loop(self) -> None:
        """Main task processing loop."""
        while self._services_running:
            try:
                # Get next task from queue
                try:
                    priority, timestamp, task_id = self.task_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Get task from pending
                with self._lock:
                    if task_id not in self.pending_tasks:
                        continue
                    
                    task = self.pending_tasks[task_id]
                    del self.pending_tasks[task_id]
                    self.running_tasks[task_id] = task
                
                # Check dependencies
                if not self._check_task_dependencies(task):
                    # Dependencies not met, requeue
                    with self._lock:
                        del self.running_tasks[task_id]
                        self.pending_tasks[task_id] = task
                        self.task_queue.put((priority, timestamp, task_id))
                    continue
                
                # Select worker node
                selected_node = self.load_balancer.select_node(task, list(self.worker_nodes.values()))
                
                if selected_node:
                    # Dispatch to remote node
                    self._dispatch_task_to_node(task, selected_node)
                else:
                    # Execute locally
                    self._execute_task_locally(task)
                
            except Exception as e:
                self.logger.error(f"Error in task processor loop: {e}")
                time.sleep(1.0)

    def _check_task_dependencies(self, task: DistributedTask) -> bool:
        """Check if task dependencies are satisfied."""
        if not task.dependencies:
            return True
        
        with self._lock:
            for dep_id in task.dependencies:
                if dep_id not in self.completed_tasks:
                    return False
        
        return True

    def _dispatch_task_to_node(self, task: DistributedTask, node: WorkerNode) -> None:
        """Dispatch task to remote worker node."""
        try:
            task.assigned_node = node.node_id
            task.started_at = time.time()
            
            # Send task to node (simplified - would use actual network protocol)
            success = self.network_manager.send_task(node, task)
            
            if not success:
                self._handle_task_failure(task, "Failed to dispatch to node")
        
        except Exception as e:
            self._handle_task_failure(task, f"Dispatch error: {e}")

    def _execute_task_locally(self, task: DistributedTask) -> None:
        """Execute task locally."""
        def execute_task():
            try:
                task.assigned_node = self.node_id
                task.started_at = time.time()
                
                # Check if function is registered
                if task.function_name not in self.function_registry:
                    raise Exception(f"Function '{task.function_name}' not registered")
                
                func = self.function_registry[task.function_name]
                
                # Execute function
                result = func(*task.args, **task.kwargs)
                
                # Handle completion
                task.result = result
                task.completed_at = time.time()
                
                with self._lock:
                    if task.task_id in self.running_tasks:
                        del self.running_tasks[task.task_id]
                    self.completed_tasks[task.task_id] = task
                    
                    # Update metrics
                    self.metrics['tasks_completed'] += 1
                    execution_time = task.completed_at - task.started_at
                    self.metrics['total_processing_time'] += execution_time
                    
                    if self.metrics['tasks_completed'] > 0:
                        self.metrics['average_task_duration'] = (
                            self.metrics['total_processing_time'] / self.metrics['tasks_completed']
                        )
                
                self.logger.info(f"Task {task.task_id} completed successfully")
                
            except Exception as e:
                self._handle_task_failure(task, str(e))
        
        # Submit to appropriate executor
        if task.cpu_requirement > 1 and self.process_pool:
            # CPU-intensive task - use process pool
            future = self.process_pool.submit(execute_task)
        else:
            # I/O or light task - use thread pool
            future = self.local_workers.submit(execute_task)

    def _handle_task_failure(self, task: DistributedTask, error: str) -> None:
        """Handle task failure with retry logic."""
        task.error = error
        task.retry_count += 1
        
        if task.retry_count <= task.max_retries:
            # Retry task
            self.logger.warning(f"Task {task.task_id} failed, retrying ({task.retry_count}/{task.max_retries}): {error}")
            
            with self._lock:
                if task.task_id in self.running_tasks:
                    del self.running_tasks[task.task_id]
                self.pending_tasks[task.task_id] = task
                
                # Requeue with lower priority
                priority_value = task.priority.value + task.retry_count
                self.task_queue.put((priority_value, time.time(), task.task_id))
        else:
            # Task failed permanently
            self.logger.error(f"Task {task.task_id} failed permanently: {error}")
            
            with self._lock:
                if task.task_id in self.running_tasks:
                    del self.running_tasks[task.task_id]
                self.failed_tasks[task.task_id] = task
                
                self.metrics['tasks_failed'] += 1

    def _heartbeat_loop(self) -> None:
        """Heartbeat loop for cluster health monitoring."""
        while self._services_running:
            try:
                current_time = time.time()
                
                # Check worker node health
                unhealthy_nodes = []
                
                with self._lock:
                    for node_id, node in self.worker_nodes.items():
                        if (current_time - node.last_heartbeat) > self.config['heartbeat_timeout']:
                            node.is_healthy = False
                            unhealthy_nodes.append(node_id)
                
                # Remove unhealthy nodes
                for node_id in unhealthy_nodes:
                    self.remove_worker_node(node_id)
                
                # Send heartbeat to other nodes (if not coordinator)
                if self.node_type != NodeType.COORDINATOR:
                    self._send_heartbeat()
                
                time.sleep(self.heartbeat_interval)
                
            except Exception as e:
                self.logger.error(f"Error in heartbeat loop: {e}")
                time.sleep(5.0)

    def _send_heartbeat(self) -> None:
        """Send heartbeat to coordinator."""
        # Simplified heartbeat - would use actual network protocol
        heartbeat_data = {
            'node_id': self.node_id,
            'timestamp': time.time(),
            'cpu_usage': self._get_cpu_usage(),
            'memory_usage': self._get_memory_usage(),
            'active_tasks': len(self.running_tasks)
        }
        
        # Send to coordinator (implementation would vary)
        self.logger.debug(f"Sending heartbeat: {heartbeat_data}")

    def _metrics_loop(self) -> None:
        """Metrics collection loop."""
        while self._services_running:
            try:
                # Update throughput metrics
                current_time = time.time()
                
                # Calculate current throughput (tasks per second)
                if hasattr(self, '_last_metrics_time'):
                    time_delta = current_time - self._last_metrics_time
                    if time_delta > 0:
                        tasks_delta = self.metrics['tasks_completed'] - getattr(self, '_last_task_count', 0)
                        current_throughput = tasks_delta / time_delta
                        self.metrics['current_throughput'] = current_throughput
                        self.metrics['peak_throughput'] = max(
                            self.metrics['peak_throughput'], current_throughput
                        )
                
                self._last_metrics_time = current_time
                self._last_task_count = self.metrics['tasks_completed']
                
                # Update resource utilization
                self.metrics['resource_utilization'] = self._calculate_resource_utilization()
                
                time.sleep(10.0)  # Update metrics every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error in metrics loop: {e}")
                time.sleep(10.0)

    def _network_service_loop(self) -> None:
        """Network service loop for handling connections."""
        while self._services_running:
            try:
                # Handle incoming connections and messages
                # This would implement actual network protocol
                time.sleep(1.0)
                
            except Exception as e:
                self.logger.error(f"Error in network service loop: {e}")
                time.sleep(1.0)

    def _reschedule_tasks_from_node(self, node_id: str) -> None:
        """Reschedule tasks from failed node."""
        tasks_to_reschedule = []
        
        with self._lock:
            # Find tasks assigned to failed node
            for task_id, task in self.running_tasks.items():
                if task.assigned_node == node_id:
                    tasks_to_reschedule.append(task)
            
            # Remove from running and add back to pending
            for task in tasks_to_reschedule:
                del self.running_tasks[task.task_id]
                task.assigned_node = None
                task.started_at = None
                self.pending_tasks[task.task_id] = task
                
                # Requeue task
                priority_value = task.priority.value
                self.task_queue.put((priority_value, time.time(), task.task_id))
        
        self.logger.info(f"Rescheduled {len(tasks_to_reschedule)} tasks from failed node {node_id}")

    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get system metrics for auto-scaling."""
        with self._lock:
            return {
                'queue_length': self.task_queue.qsize(),
                'running_tasks': len(self.running_tasks),
                'worker_count': len(self.worker_nodes),
                'average_cpu_usage': sum(n.cpu_usage for n in self.worker_nodes.values()) / max(len(self.worker_nodes), 1),
                'average_memory_usage': sum(n.memory_usage for n in self.worker_nodes.values()) / max(len(self.worker_nodes), 1),
                'current_throughput': self.metrics['current_throughput']
            }

    def _calculate_resource_utilization(self) -> float:
        """Calculate overall resource utilization."""
        if not self.worker_nodes:
            return 0.0
        
        with self._lock:
            total_utilization = 0.0
            
            for node in self.worker_nodes.values():
                cpu_util = node.cpu_usage / 100.0
                memory_util = node.memory_usage / 100.0
                task_util = node.active_tasks / max(node.cpu_cores, 1)
                
                node_utilization = (cpu_util + memory_util + task_util) / 3.0
                total_utilization += node_utilization
            
            return total_utilization / len(self.worker_nodes)

    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        try:
            import psutil
            return psutil.cpu_percent()
        except ImportError:
            # Fallback if psutil not available
            return 50.0

    def _get_memory_usage(self) -> float:
        """Get current memory usage percentage."""
        try:
            import psutil
            return psutil.virtual_memory().percent
        except ImportError:
            # Fallback if psutil not available
            return 60.0


class LoadBalancer:
    """Intelligent load balancer for task distribution."""
    
    def __init__(self):
        self.algorithm = "weighted_round_robin"
        self.round_robin_index = 0
        
    def select_node(self, task: DistributedTask, nodes: List[WorkerNode]) -> Optional[WorkerNode]:
        """
        Select optimal node for task execution.
        
        Args:
            task: Task to be executed
            nodes: Available worker nodes
            
        Returns:
            Selected worker node or None if no suitable node
        """
        # Filter nodes that can handle the task
        suitable_nodes = [node for node in nodes if node.is_healthy and node.can_handle_task(task)]
        
        if not suitable_nodes:
            return None
        
        if self.algorithm == "weighted_round_robin":
            return self._weighted_round_robin(suitable_nodes)
        elif self.algorithm == "least_loaded":
            return self._least_loaded(suitable_nodes)
        elif self.algorithm == "performance_based":
            return self._performance_based(suitable_nodes)
        else:
            return suitable_nodes[0]  # Fallback
    
    def _weighted_round_robin(self, nodes: List[WorkerNode]) -> WorkerNode:
        """Weighted round-robin selection."""
        # Weight based on inverse load score
        weights = [1.0 / (node.get_load_score() + 0.1) for node in nodes]
        total_weight = sum(weights)
        
        # Normalize weights
        weights = [w / total_weight for w in weights]
        
        # Select based on weights (simplified)
        import random
        return random.choices(nodes, weights=weights)[0]
    
    def _least_loaded(self, nodes: List[WorkerNode]) -> WorkerNode:
        """Select least loaded node."""
        return min(nodes, key=lambda node: node.get_load_score())
    
    def _performance_based(self, nodes: List[WorkerNode]) -> WorkerNode:
        """Select node based on performance history."""
        # Calculate performance score
        best_node = nodes[0]
        best_score = 0.0
        
        for node in nodes:
            # Performance score based on tasks completed and average duration
            if node.tasks_completed > 0:
                success_rate = node.tasks_completed / (node.tasks_completed + node.tasks_failed)
                speed_score = 1.0 / (node.average_task_duration + 0.1)
                load_score = 1.0 / (node.get_load_score() + 0.1)
                
                performance_score = success_rate * speed_score * load_score
                
                if performance_score > best_score:
                    best_score = performance_score
                    best_node = node
        
        return best_node


class TaskDispatcher:
    """Task dispatcher for network communication."""
    
    def __init__(self):
        self.compression_enabled = True
        self.encryption_enabled = False
    
    def dispatch_task(self, task: DistributedTask, node: WorkerNode) -> bool:
        """
        Dispatch task to worker node.
        
        Args:
            task: Task to dispatch
            node: Target worker node
            
        Returns:
            True if successful dispatch
        """
        try:
            # Serialize task
            task_data = task.serialize()
            
            # Compress if enabled
            if self.compression_enabled:
                task_data = self._compress_data(task_data)
            
            # Encrypt if enabled
            if self.encryption_enabled:
                task_data = self._encrypt_data(task_data)
            
            # Send over network (simplified)
            return self._send_to_node(task_data, node)
            
        except Exception as e:
            logging.error(f"Failed to dispatch task {task.task_id} to node {node.node_id}: {e}")
            return False
    
    def _compress_data(self, data: bytes) -> bytes:
        """Compress data for network transmission."""
        import gzip
        return gzip.compress(data)
    
    def _encrypt_data(self, data: bytes) -> bytes:
        """Encrypt data for secure transmission."""
        # Placeholder for encryption
        return data
    
    def _send_to_node(self, data: bytes, node: WorkerNode) -> bool:
        """Send data to worker node."""
        # Placeholder for network transmission
        return True


class AutoScaler:
    """Auto-scaling system for dynamic resource management."""
    
    def __init__(self, min_workers: int = 2, max_workers: int = 100, 
                 scaling_strategy: ScalingStrategy = ScalingStrategy.HYBRID):
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.scaling_strategy = scaling_strategy
        
        # Scaling thresholds
        self.scale_up_threshold = 0.8  # 80% utilization
        self.scale_down_threshold = 0.3  # 30% utilization
        self.scale_up_queue_threshold = 10  # Tasks in queue
        self.scale_down_queue_threshold = 2
        
        # Cooldown periods
        self.scale_up_cooldown = 60.0  # 1 minute
        self.scale_down_cooldown = 300.0  # 5 minutes
        
        self.last_scale_up = 0.0
        self.last_scale_down = 0.0
    
    def evaluate_scaling_need(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate if scaling is needed.
        
        Args:
            metrics: Current system metrics
            
        Returns:
            Scaling recommendation
        """
        current_time = time.time()
        recommendation = {
            'action': 'none',
            'reason': '',
            'target_workers': metrics['worker_count']
        }
        
        if self.scaling_strategy == ScalingStrategy.CPU_BASED:
            recommendation = self._cpu_based_scaling(metrics, current_time)
        elif self.scaling_strategy == ScalingStrategy.QUEUE_LENGTH:
            recommendation = self._queue_based_scaling(metrics, current_time)
        elif self.scaling_strategy == ScalingStrategy.HYBRID:
            recommendation = self._hybrid_scaling(metrics, current_time)
        
        return recommendation
    
    def _cpu_based_scaling(self, metrics: Dict[str, Any], current_time: float) -> Dict[str, Any]:
        """CPU-based scaling logic."""
        cpu_usage = metrics['average_cpu_usage']
        current_workers = metrics['worker_count']
        
        if (cpu_usage > self.scale_up_threshold * 100 and 
            current_workers < self.max_workers and
            (current_time - self.last_scale_up) > self.scale_up_cooldown):
            
            return {
                'action': 'scale_up',
                'reason': f'High CPU usage: {cpu_usage:.1f}%',
                'target_workers': min(current_workers + 1, self.max_workers)
            }
        
        elif (cpu_usage < self.scale_down_threshold * 100 and 
              current_workers > self.min_workers and
              (current_time - self.last_scale_down) > self.scale_down_cooldown):
            
            return {
                'action': 'scale_down',
                'reason': f'Low CPU usage: {cpu_usage:.1f}%',
                'target_workers': max(current_workers - 1, self.min_workers)
            }
        
        return {'action': 'none', 'reason': 'CPU usage within normal range', 'target_workers': current_workers}
    
    def _queue_based_scaling(self, metrics: Dict[str, Any], current_time: float) -> Dict[str, Any]:
        """Queue length-based scaling logic."""
        queue_length = metrics['queue_length']
        current_workers = metrics['worker_count']
        
        if (queue_length > self.scale_up_queue_threshold and 
            current_workers < self.max_workers and
            (current_time - self.last_scale_up) > self.scale_up_cooldown):
            
            return {
                'action': 'scale_up',
                'reason': f'High queue length: {queue_length}',
                'target_workers': min(current_workers + 1, self.max_workers)
            }
        
        elif (queue_length < self.scale_down_queue_threshold and 
              current_workers > self.min_workers and
              (current_time - self.last_scale_down) > self.scale_down_cooldown):
            
            return {
                'action': 'scale_down',
                'reason': f'Low queue length: {queue_length}',
                'target_workers': max(current_workers - 1, self.min_workers)
            }
        
        return {'action': 'none', 'reason': 'Queue length within normal range', 'target_workers': current_workers}
    
    def _hybrid_scaling(self, metrics: Dict[str, Any], current_time: float) -> Dict[str, Any]:
        """Hybrid scaling combining multiple factors."""
        cpu_usage = metrics['average_cpu_usage']
        queue_length = metrics['queue_length']
        current_workers = metrics['worker_count']
        throughput = metrics['current_throughput']
        
        # Calculate scaling score
        cpu_score = (cpu_usage / 100.0 - 0.5) * 2  # -1 to 1
        queue_score = min(queue_length / 20.0, 1.0)  # 0 to 1
        throughput_score = min(throughput / 10.0, 1.0)  # 0 to 1
        
        scaling_score = (cpu_score + queue_score + throughput_score) / 3.0
        
        if (scaling_score > 0.6 and 
            current_workers < self.max_workers and
            (current_time - self.last_scale_up) > self.scale_up_cooldown):
            
            return {
                'action': 'scale_up',
                'reason': f'High scaling score: {scaling_score:.2f}',
                'target_workers': min(current_workers + 1, self.max_workers)
            }
        
        elif (scaling_score < -0.3 and 
              current_workers > self.min_workers and
              (current_time - self.last_scale_down) > self.scale_down_cooldown):
            
            return {
                'action': 'scale_down',
                'reason': f'Low scaling score: {scaling_score:.2f}',
                'target_workers': max(current_workers - 1, self.min_workers)
            }
        
        return {'action': 'none', 'reason': f'Scaling score normal: {scaling_score:.2f}', 'target_workers': current_workers}


class DistributedCacheManager:
    """Distributed caching system for performance optimization."""
    
    def __init__(self, cache_size: int = 1000):
        self.cache_size = cache_size
        self.local_cache: Dict[str, Any] = {}
        self.cache_access_times: Dict[str, float] = {}
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key in self.local_cache:
            self.cache_access_times[key] = time.time()
            return self.local_cache[key]
        return None
    
    def put(self, key: str, value: Any) -> None:
        """Put value in cache."""
        # Evict if cache is full
        if len(self.local_cache) >= self.cache_size:
            self._evict_lru()
        
        self.local_cache[key] = value
        self.cache_access_times[key] = time.time()
    
    def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if not self.cache_access_times:
            return
        
        lru_key = min(self.cache_access_times.items(), key=lambda x: x[1])[0]
        del self.local_cache[lru_key]
        del self.cache_access_times[lru_key]


class NetworkManager:
    """Network communication manager."""
    
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.connections: Dict[str, Any] = {}
    
    def send_task(self, node: WorkerNode, task: DistributedTask) -> bool:
        """Send task to worker node."""
        try:
            # Simplified network communication
            # In production, would use actual sockets, HTTP, or message queues
            self._log_network_operation("send", len(task.serialize()))
            return True
        except Exception as e:
            logging.error(f"Failed to send task to {node.node_id}: {e}")
            return False
    
    def _log_network_operation(self, operation: str, bytes_transferred: int) -> None:
        """Log network operation for metrics."""
        # Would update network metrics
        pass