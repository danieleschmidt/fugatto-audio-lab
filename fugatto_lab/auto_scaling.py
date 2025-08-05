"""Auto-Scaling and Load Balancing System for Fugatto Audio Lab.

Advanced auto-scaling capabilities with predictive scaling, load balancing,
and resource management for high-performance audio processing workloads.
"""

import asyncio
import time
import json
import logging
import math
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import threading
import weakref
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import heapq

logger = logging.getLogger(__name__)


class ScalingPolicy(Enum):
    """Auto-scaling policies."""
    REACTIVE = "reactive"           # Scale based on current metrics
    PREDICTIVE = "predictive"       # Scale based on predicted load
    SCHEDULED = "scheduled"         # Scale based on schedule
    HYBRID = "hybrid"              # Combination of policies


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"     # Distribute requests evenly
    LEAST_CONNECTIONS = "least_connections"  # Route to least busy worker
    LEAST_RESPONSE_TIME = "least_response_time"  # Route to fastest worker
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"  # Weighted distribution
    RESOURCE_BASED = "resource_based"  # Route based on resource usage
    INTELLIGENT = "intelligent"     # ML-based routing


class WorkerState(Enum):
    """Worker instance states."""
    INITIALIZING = "initializing"
    HEALTHY = "healthy"
    BUSY = "busy"
    OVERLOADED = "overloaded"
    UNHEALTHY = "unhealthy"
    TERMINATING = "terminating"


@dataclass
class WorkerInstance:
    """Represents a worker instance for processing."""
    
    worker_id: str
    state: WorkerState = WorkerState.INITIALIZING
    capacity: int = 10  # Maximum concurrent tasks
    current_load: int = 0  # Current active tasks
    total_processed: int = 0
    total_errors: int = 0
    average_response_time: float = 0.0
    last_health_check: float = field(default_factory=time.time)
    created_at: float = field(default_factory=time.time)
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    gpu_usage: float = 0.0
    
    @property
    def utilization(self) -> float:
        """Calculate current utilization percentage."""
        return (self.current_load / self.capacity) * 100 if self.capacity > 0 else 0
    
    @property
    def error_rate(self) -> float:
        """Calculate error rate percentage."""
        total = self.total_processed + self.total_errors
        return (self.total_errors / total) * 100 if total > 0 else 0
    
    @property
    def is_healthy(self) -> bool:
        """Check if worker is healthy."""
        return (self.state in [WorkerState.HEALTHY, WorkerState.BUSY] and
                self.error_rate < 10.0 and  # Less than 10% error rate
                time.time() - self.last_health_check < 60.0)  # Recent health check
    
    def can_accept_task(self) -> bool:
        """Check if worker can accept new task."""
        return (self.is_healthy and 
                self.current_load < self.capacity and
                self.state != WorkerState.OVERLOADED)
    
    def update_metrics(self, response_time: float, success: bool, 
                      cpu_usage: float = None, memory_usage: float = None, 
                      gpu_usage: float = None):
        """Update worker metrics."""
        # Update response time (exponential moving average)
        alpha = 0.1
        self.average_response_time = (alpha * response_time + 
                                    (1 - alpha) * self.average_response_time)
        
        # Update counters
        if success:
            self.total_processed += 1
        else:
            self.total_errors += 1
        
        # Update resource usage
        if cpu_usage is not None:
            self.cpu_usage = cpu_usage
        if memory_usage is not None:
            self.memory_usage = memory_usage
        if gpu_usage is not None:
            self.gpu_usage = gpu_usage
        
        # Update state based on load
        if self.utilization > 90:
            self.state = WorkerState.OVERLOADED
        elif self.utilization > 70:
            self.state = WorkerState.BUSY
        elif self.is_healthy:
            self.state = WorkerState.HEALTHY
        
        self.last_health_check = time.time()


@dataclass
class ScalingMetrics:
    """Metrics for scaling decisions."""
    
    timestamp: float = field(default_factory=time.time)
    total_workers: int = 0
    healthy_workers: int = 0
    average_utilization: float = 0.0
    average_response_time: float = 0.0
    queue_length: int = 0
    requests_per_second: float = 0.0
    error_rate: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    gpu_usage: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "total_workers": self.total_workers,
            "healthy_workers": self.healthy_workers,
            "average_utilization": self.average_utilization,
            "average_response_time": self.average_response_time,
            "queue_length": self.queue_length,
            "requests_per_second": self.requests_per_second,
            "error_rate": self.error_rate,
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "gpu_usage": self.gpu_usage
        }


class LoadBalancer:
    """Intelligent load balancer for distributing work."""
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.INTELLIGENT):
        self.strategy = strategy
        self.workers = {}  # worker_id -> WorkerInstance
        self.round_robin_index = 0
        self.routing_history = deque(maxlen=1000)
        
        # ML-based routing (simplified)
        self.routing_model = None
        self.feature_weights = {
            'utilization': 0.4,
            'response_time': 0.3,
            'error_rate': 0.2,
            'resource_usage': 0.1
        }
        
        logger.info(f"LoadBalancer initialized with {strategy.value} strategy")
    
    def add_worker(self, worker: WorkerInstance):
        """Add worker to load balancer."""
        self.workers[worker.worker_id] = worker
        logger.info(f"Added worker {worker.worker_id} to load balancer")
    
    def remove_worker(self, worker_id: str):
        """Remove worker from load balancer."""
        if worker_id in self.workers:
            del self.workers[worker_id]
            logger.info(f"Removed worker {worker_id} from load balancer")
    
    def select_worker(self, task_context: Dict[str, Any] = None) -> Optional[WorkerInstance]:
        """Select best worker for task."""
        available_workers = [w for w in self.workers.values() if w.can_accept_task()]
        
        if not available_workers:
            return None
        
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._select_round_robin(available_workers)
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._select_least_connections(available_workers)
        elif self.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
            return self._select_least_response_time(available_workers)
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._select_weighted_round_robin(available_workers)
        elif self.strategy == LoadBalancingStrategy.RESOURCE_BASED:
            return self._select_resource_based(available_workers)
        elif self.strategy == LoadBalancingStrategy.INTELLIGENT:
            return self._select_intelligent(available_workers, task_context)
        else:
            return available_workers[0]  # Fallback
    
    def _select_round_robin(self, workers: List[WorkerInstance]) -> WorkerInstance:
        """Round-robin selection."""
        if not workers:
            return None
        
        worker = workers[self.round_robin_index % len(workers)]
        self.round_robin_index += 1
        return worker
    
    def _select_least_connections(self, workers: List[WorkerInstance]) -> WorkerInstance:
        """Select worker with least connections."""
        return min(workers, key=lambda w: w.current_load)
    
    def _select_least_response_time(self, workers: List[WorkerInstance]) -> WorkerInstance:
        """Select worker with lowest response time."""
        return min(workers, key=lambda w: w.average_response_time)
    
    def _select_weighted_round_robin(self, workers: List[WorkerInstance]) -> WorkerInstance:
        """Select worker using weighted round-robin based on capacity."""
        if not workers:
            return None
        
        # Create weighted list based on available capacity
        weighted_workers = []
        for worker in workers:
            available_capacity = worker.capacity - worker.current_load
            weighted_workers.extend([worker] * max(1, available_capacity))
        
        if weighted_workers:
            worker = weighted_workers[self.round_robin_index % len(weighted_workers)]
            self.round_robin_index += 1
            return worker
        
        return workers[0]
    
    def _select_resource_based(self, workers: List[WorkerInstance]) -> WorkerInstance:
        """Select worker based on resource availability."""
        def resource_score(worker):
            # Lower score is better (more available resources)
            cpu_score = worker.cpu_usage / 100.0
            memory_score = worker.memory_usage / 100.0
            gpu_score = worker.gpu_usage / 100.0 if worker.gpu_usage > 0 else 0
            utilization_score = worker.utilization / 100.0
            
            return cpu_score + memory_score + gpu_score + utilization_score
        
        return min(workers, key=resource_score)
    
    def _select_intelligent(self, workers: List[WorkerInstance], 
                           task_context: Dict[str, Any] = None) -> WorkerInstance:
        """Intelligent ML-based worker selection."""
        def calculate_worker_score(worker):
            # Calculate composite score based on multiple factors
            utilization_score = (100 - worker.utilization) / 100.0  # Higher available capacity = better
            response_time_score = 1.0 / (1.0 + worker.average_response_time)  # Lower response time = better
            error_rate_score = (100 - worker.error_rate) / 100.0  # Lower error rate = better
            
            # Resource availability score
            resource_score = (3 - (worker.cpu_usage + worker.memory_usage + worker.gpu_usage) / 100.0) / 3.0
            
            # Weighted composite score
            composite_score = (
                self.feature_weights['utilization'] * utilization_score +
                self.feature_weights['response_time'] * response_time_score +
                self.feature_weights['error_rate'] * error_rate_score +
                self.feature_weights['resource_usage'] * resource_score
            )
            
            return composite_score
        
        # Select worker with highest score
        best_worker = max(workers, key=calculate_worker_score)
        
        # Record routing decision for learning
        self.routing_history.append({
            'worker_id': best_worker.worker_id,
            'timestamp': time.time(),
            'worker_score': calculate_worker_score(best_worker),
            'task_context': task_context
        })
        
        return best_worker
    
    def get_load_balancer_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        healthy_workers = [w for w in self.workers.values() if w.is_healthy]
        
        if healthy_workers:
            avg_utilization = sum(w.utilization for w in healthy_workers) / len(healthy_workers)
            avg_response_time = sum(w.average_response_time for w in healthy_workers) / len(healthy_workers)
            avg_error_rate = sum(w.error_rate for w in healthy_workers) / len(healthy_workers)
        else:
            avg_utilization = avg_response_time = avg_error_rate = 0.0
        
        return {
            "strategy": self.strategy.value,
            "total_workers": len(self.workers),
            "healthy_workers": len(healthy_workers),
            "average_utilization": avg_utilization,
            "average_response_time": avg_response_time,
            "average_error_rate": avg_error_rate,
            "routing_decisions": len(self.routing_history)
        }


class PredictiveScaler:
    """Predictive scaling based on historical patterns."""
    
    def __init__(self, history_window: int = 1440):  # 24 hours of minutes
        self.history_window = history_window
        self.metrics_history = deque(maxlen=history_window)
        self.seasonal_patterns = {}
        self.trend_coefficients = {'slope': 0.0, 'intercept': 0.0}
        
        logger.info("PredictiveScaler initialized")
    
    def add_metrics(self, metrics: ScalingMetrics):
        """Add metrics to history for prediction."""
        self.metrics_history.append(metrics)
        
        # Update patterns periodically
        if len(self.metrics_history) % 60 == 0:  # Every hour
            self._update_patterns()
    
    def predict_load(self, minutes_ahead: int = 30) -> float:
        """Predict load N minutes ahead."""
        if len(self.metrics_history) < 10:
            # Not enough history, return current load
            return self.metrics_history[-1].requests_per_second if self.metrics_history else 0.0
        
        current_time = time.time()
        future_time = current_time + (minutes_ahead * 60)
        
        # Get seasonal component
        seasonal_load = self._get_seasonal_prediction(future_time)
        
        # Get trend component
        trend_load = self._get_trend_prediction(future_time)
        
        # Combine predictions
        predicted_load = max(0.0, seasonal_load + trend_load)
        
        logger.debug(f"Predicted load in {minutes_ahead}min: {predicted_load:.2f} RPS")
        
        return predicted_load
    
    def predict_required_workers(self, minutes_ahead: int = 30, 
                               worker_capacity: float = 10.0) -> int:
        """Predict number of workers needed."""
        predicted_load = self.predict_load(minutes_ahead)
        
        # Add safety margin
        safety_margin = 1.2
        required_capacity = predicted_load * safety_margin
        
        required_workers = math.ceil(required_capacity / worker_capacity)
        
        return max(1, required_workers)  # At least 1 worker
    
    def _update_patterns(self):
        """Update seasonal patterns and trends."""
        if len(self.metrics_history) < 60:
            return
        
        # Extract time series data
        timestamps = [m.timestamp for m in self.metrics_history]
        loads = [m.requests_per_second for m in self.metrics_history]
        
        # Update trend using simple linear regression
        self._update_trend(timestamps, loads)
        
        # Update seasonal patterns
        self._update_seasonal_patterns(timestamps, loads)
    
    def _update_trend(self, timestamps: List[float], loads: List[float]):
        """Update linear trend coefficients."""
        if len(timestamps) < 2:
            return
        
        # Simple linear regression
        n = len(timestamps)
        sum_x = sum(timestamps)
        sum_y = sum(loads)
        sum_xy = sum(x * y for x, y in zip(timestamps, loads))
        sum_x2 = sum(x * x for x in timestamps)
        
        # Calculate slope and intercept
        denominator = n * sum_x2 - sum_x * sum_x
        if denominator != 0:
            slope = (n * sum_xy - sum_x * sum_y) / denominator
            intercept = (sum_y - slope * sum_x) / n
            
            # Use exponential smoothing to update coefficients
            alpha = 0.1
            self.trend_coefficients['slope'] = (alpha * slope + 
                                              (1 - alpha) * self.trend_coefficients['slope'])
            self.trend_coefficients['intercept'] = (alpha * intercept + 
                                                   (1 - alpha) * self.trend_coefficients['intercept'])
    
    def _update_seasonal_patterns(self, timestamps: List[float], loads: List[float]):
        """Update seasonal patterns (hourly, daily, weekly)."""
        import datetime
        
        # Group by hour of day
        hourly_loads = defaultdict(list)
        daily_loads = defaultdict(list)
        
        for timestamp, load in zip(timestamps, loads):
            dt = datetime.datetime.fromtimestamp(timestamp)
            hour_key = dt.hour
            day_key = dt.weekday()  # 0 = Monday
            
            hourly_loads[hour_key].append(load)
            daily_loads[day_key].append(load)
        
        # Calculate average loads for each pattern
        self.seasonal_patterns['hourly'] = {
            hour: sum(loads) / len(loads) 
            for hour, loads in hourly_loads.items()
        }
        
        self.seasonal_patterns['daily'] = {
            day: sum(loads) / len(loads) 
            for day, loads in daily_loads.items()
        }
    
    def _get_seasonal_prediction(self, future_timestamp: float) -> float:
        """Get seasonal component of prediction."""
        import datetime
        
        dt = datetime.datetime.fromtimestamp(future_timestamp)
        hour = dt.hour
        day = dt.weekday()
        
        # Get seasonal components
        hourly_component = self.seasonal_patterns.get('hourly', {}).get(hour, 0.0)
        daily_component = self.seasonal_patterns.get('daily', {}).get(day, 0.0)
        
        # Combine seasonal components (weighted average)
        seasonal_load = 0.7 * hourly_component + 0.3 * daily_component
        
        return seasonal_load
    
    def _get_trend_prediction(self, future_timestamp: float) -> float:
        """Get trend component of prediction."""
        slope = self.trend_coefficients['slope']
        intercept = self.trend_coefficients['intercept']
        
        return slope * future_timestamp + intercept


class AutoScaler:
    """Main auto-scaling orchestrator."""
    
    def __init__(self, 
                 min_workers: int = 1,
                 max_workers: int = 20,
                 target_utilization: float = 70.0,
                 scale_up_threshold: float = 80.0,
                 scale_down_threshold: float = 50.0,
                 cooldown_period: int = 300,  # 5 minutes
                 policy: ScalingPolicy = ScalingPolicy.HYBRID):
        
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.target_utilization = target_utilization
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.cooldown_period = cooldown_period
        self.policy = policy
        
        # Components
        self.load_balancer = LoadBalancer(LoadBalancingStrategy.INTELLIGENT)
        self.predictive_scaler = PredictiveScaler()
        
        # State tracking
        self.last_scaling_action = 0
        self.scaling_history = deque(maxlen=1000)
        self.current_metrics = None
        
        # Task queue
        self.task_queue = asyncio.Queue()
        self.processing_tasks = {}  # task_id -> (worker, start_time)
        
        # Background monitoring
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info(f"AutoScaler initialized: {min_workers}-{max_workers} workers, "
                   f"{policy.value} policy, {target_utilization}% target utilization")
    
    def add_worker(self, worker_id: str, capacity: int = 10) -> WorkerInstance:
        """Add new worker instance."""
        worker = WorkerInstance(
            worker_id=worker_id,
            capacity=capacity,
            state=WorkerState.INITIALIZING
        )
        
        self.load_balancer.add_worker(worker)
        
        # Simulate worker initialization
        def initialize_worker():
            time.sleep(2)  # Simulate startup time
            worker.state = WorkerState.HEALTHY
            logger.info(f"Worker {worker_id} initialized and ready")
        
        thread = threading.Thread(target=initialize_worker, daemon=True)
        thread.start()
        
        return worker
    
    def remove_worker(self, worker_id: str):
        """Remove worker instance."""
        if worker_id in self.load_balancer.workers:
            worker = self.load_balancer.workers[worker_id]
            worker.state = WorkerState.TERMINATING
            
            # Wait for current tasks to complete (simplified)
            def terminate_worker():
                time.sleep(5)  # Grace period
                self.load_balancer.remove_worker(worker_id)
                logger.info(f"Worker {worker_id} terminated")
            
            thread = threading.Thread(target=terminate_worker, daemon=True)
            thread.start()
    
    async def process_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process task with auto-scaling."""
        task_id = task_data.get('id', f"task_{int(time.time() * 1000)}")
        
        # Select worker
        worker = self.load_balancer.select_worker(task_data)
        
        if not worker:
            # No available workers, queue task
            await self.task_queue.put((task_id, task_data))
            logger.warning(f"Task {task_id} queued - no available workers")
            return {"status": "queued", "task_id": task_id}
        
        # Process task
        start_time = time.time()
        worker.current_load += 1
        self.processing_tasks[task_id] = (worker, start_time)
        
        try:
            # Simulate task processing
            processing_time = task_data.get('processing_time', 2.0)
            await asyncio.sleep(processing_time)
            
            # Task completed successfully
            duration = time.time() - start_time
            worker.update_metrics(duration, success=True)
            
            result = {
                "status": "completed",
                "task_id": task_id,
                "worker_id": worker.worker_id,
                "processing_time": duration
            }
            
        except Exception as e:
            # Task failed
            duration = time.time() - start_time
            worker.update_metrics(duration, success=False)
            
            result = {
                "status": "failed",
                "task_id": task_id,
                "worker_id": worker.worker_id,
                "error": str(e)
            }
            
            logger.error(f"Task {task_id} failed on worker {worker.worker_id}: {e}")
        
        finally:
            # Clean up
            worker.current_load = max(0, worker.current_load - 1)
            if task_id in self.processing_tasks:
                del self.processing_tasks[task_id]
        
        return result
    
    def _monitor_loop(self):
        """Background monitoring and scaling loop."""
        while self.running:
            try:
                time.sleep(30)  # Check every 30 seconds
                
                # Collect current metrics
                self.current_metrics = self._collect_metrics()
                
                # Add to predictive scaler
                self.predictive_scaler.add_metrics(self.current_metrics)
                
                # Make scaling decision
                self._make_scaling_decision()
                
            except Exception as e:
                logger.error(f"Auto-scaler monitor error: {e}")
    
    def _collect_metrics(self) -> ScalingMetrics:
        """Collect current system metrics."""
        workers = list(self.load_balancer.workers.values())
        healthy_workers = [w for w in workers if w.is_healthy]
        
        if healthy_workers:
            avg_utilization = sum(w.utilization for w in healthy_workers) / len(healthy_workers)
            avg_response_time = sum(w.average_response_time for w in healthy_workers) / len(healthy_workers)
            avg_error_rate = sum(w.error_rate for w in healthy_workers) / len(healthy_workers)
            avg_cpu = sum(w.cpu_usage for w in healthy_workers) / len(healthy_workers)
            avg_memory = sum(w.memory_usage for w in healthy_workers) / len(healthy_workers)
            avg_gpu = sum(w.gpu_usage for w in healthy_workers) / len(healthy_workers)
        else:
            avg_utilization = avg_response_time = avg_error_rate = 0.0
            avg_cpu = avg_memory = avg_gpu = 0.0
        
        # Calculate requests per second (simplified)
        current_time = time.time()
        recent_completions = sum(
            1 for worker in healthy_workers
            for _ in range(worker.total_processed)
            if current_time - worker.last_health_check < 60  # Last minute
        )
        requests_per_second = recent_completions / 60.0
        
        return ScalingMetrics(
            total_workers=len(workers),
            healthy_workers=len(healthy_workers),
            average_utilization=avg_utilization,
            average_response_time=avg_response_time,
            queue_length=self.task_queue.qsize(),
            requests_per_second=requests_per_second,
            error_rate=avg_error_rate,
            cpu_usage=avg_cpu,
            memory_usage=avg_memory,
            gpu_usage=avg_gpu
        )
    
    def _make_scaling_decision(self):
        """Make scaling decision based on policy."""
        if not self.current_metrics:
            return
        
        current_time = time.time()
        
        # Check cooldown period
        if current_time - self.last_scaling_action < self.cooldown_period:
            return
        
        if self.policy == ScalingPolicy.REACTIVE:
            self._reactive_scaling()
        elif self.policy == ScalingPolicy.PREDICTIVE:
            self._predictive_scaling()
        elif self.policy == ScalingPolicy.SCHEDULED:
            self._scheduled_scaling()
        elif self.policy == ScalingPolicy.HYBRID:
            self._hybrid_scaling()
    
    def _reactive_scaling(self):
        """Reactive scaling based on current metrics."""
        metrics = self.current_metrics
        
        # Scale up conditions
        if (metrics.average_utilization > self.scale_up_threshold or
            metrics.queue_length > 10 or
            metrics.average_response_time > 5.0):
            
            if metrics.healthy_workers < self.max_workers:
                self._scale_up(1)
        
        # Scale down conditions
        elif (metrics.average_utilization < self.scale_down_threshold and
              metrics.queue_length == 0 and
              metrics.healthy_workers > self.min_workers):
            
            self._scale_down(1)
    
    def _predictive_scaling(self):
        """Predictive scaling based on forecasted load."""
        predicted_workers = self.predictive_scaler.predict_required_workers(
            minutes_ahead=15  # Scale for 15 minutes ahead
        )
        
        current_healthy = self.current_metrics.healthy_workers
        
        if predicted_workers > current_healthy and current_healthy < self.max_workers:
            scale_up_count = min(predicted_workers - current_healthy, 
                               self.max_workers - current_healthy)
            self._scale_up(scale_up_count)
        
        elif predicted_workers < current_healthy and current_healthy > self.min_workers:
            scale_down_count = min(current_healthy - predicted_workers,
                                 current_healthy - self.min_workers)
            self._scale_down(scale_down_count)
    
    def _scheduled_scaling(self):
        """Scheduled scaling based on time patterns."""
        import datetime
        
        now = datetime.datetime.now()
        hour = now.hour
        day = now.weekday()  # 0 = Monday
        
        # Define scheduled scaling rules (example)
        if day < 5:  # Weekdays
            if 9 <= hour <= 17:  # Business hours
                target_workers = max(3, self.min_workers)
            else:
                target_workers = self.min_workers
        else:  # Weekends
            target_workers = self.min_workers
        
        current_healthy = self.current_metrics.healthy_workers
        
        if target_workers > current_healthy:
            self._scale_up(target_workers - current_healthy)
        elif target_workers < current_healthy:
            self._scale_down(current_healthy - target_workers)
    
    def _hybrid_scaling(self):
        """Hybrid scaling combining multiple strategies."""
        # Use predictive scaling as primary
        self._predictive_scaling()
        
        # Override with reactive scaling if needed
        metrics = self.current_metrics
        
        # Emergency scale up
        if (metrics.average_utilization > 95 or
            metrics.queue_length > 20 or
            metrics.average_response_time > 10.0):
            
            if metrics.healthy_workers < self.max_workers:
                self._scale_up(2)  # Aggressive scale up
        
        # Emergency scale down (cost optimization)
        elif (metrics.average_utilization < 20 and
              metrics.queue_length == 0 and
              metrics.healthy_workers > self.min_workers + 1):
            
            self._scale_down(1)
    
    def _scale_up(self, count: int):
        """Scale up by adding workers."""
        current_workers = len(self.load_balancer.workers)
        actual_count = min(count, self.max_workers - current_workers)
        
        if actual_count <= 0:
            return
        
        for i in range(actual_count):
            worker_id = f"worker_{int(time.time() * 1000)}_{i}"
            self.add_worker(worker_id)
        
        self.last_scaling_action = time.time()
        
        # Record scaling action
        self.scaling_history.append({
            'timestamp': time.time(),
            'action': 'scale_up',
            'count': actual_count,
            'total_workers': current_workers + actual_count,
            'trigger_metrics': self.current_metrics.to_dict()
        })
        
        logger.info(f"Scaled up: added {actual_count} workers "
                   f"(total: {current_workers + actual_count})")
    
    def _scale_down(self, count: int):
        """Scale down by removing workers."""
        current_workers = len(self.load_balancer.workers)
        actual_count = min(count, current_workers - self.min_workers)
        
        if actual_count <= 0:
            return
        
        # Select workers to remove (prefer least utilized)
        workers_to_remove = sorted(
            self.load_balancer.workers.values(),
            key=lambda w: (w.current_load, w.utilization)
        )[:actual_count]
        
        for worker in workers_to_remove:
            self.remove_worker(worker.worker_id)
        
        self.last_scaling_action = time.time()
        
        # Record scaling action
        self.scaling_history.append({
            'timestamp': time.time(),
            'action': 'scale_down',
            'count': actual_count,
            'total_workers': current_workers - actual_count,
            'trigger_metrics': self.current_metrics.to_dict()
        })
        
        logger.info(f"Scaled down: removed {actual_count} workers "
                   f"(total: {current_workers - actual_count})")
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get comprehensive scaling status."""
        return {
            "policy": self.policy.value,
            "min_workers": self.min_workers,
            "max_workers": self.max_workers,
            "target_utilization": self.target_utilization,
            "current_metrics": self.current_metrics.to_dict() if self.current_metrics else {},
            "load_balancer_stats": self.load_balancer.get_load_balancer_stats(),
            "scaling_history": list(self.scaling_history)[-10:],  # Recent history
            "predicted_load": self.predictive_scaler.predict_load(30),
            "predicted_workers": self.predictive_scaler.predict_required_workers(30)
        }
    
    def save_scaling_report(self, filepath: str):
        """Save comprehensive scaling report."""
        report = {
            "timestamp": time.time(),
            "scaling_status": self.get_scaling_status(),
            "configuration": {
                "min_workers": self.min_workers,
                "max_workers": self.max_workers,
                "target_utilization": self.target_utilization,
                "scale_up_threshold": self.scale_up_threshold,
                "scale_down_threshold": self.scale_down_threshold,
                "cooldown_period": self.cooldown_period,
                "policy": self.policy.value
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Scaling report saved to: {filepath}")
    
    def shutdown(self):
        """Shutdown auto-scaler."""
        self.running = False
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=10.0)
        
        # Terminate all workers
        for worker_id in list(self.load_balancer.workers.keys()):
            self.remove_worker(worker_id)


# Global auto-scaler instance
_global_auto_scaler = None

def get_global_auto_scaler() -> AutoScaler:
    """Get global auto-scaler instance."""
    global _global_auto_scaler
    if _global_auto_scaler is None:
        _global_auto_scaler = AutoScaler()
    return _global_auto_scaler


# Convenience functions

async def process_audio_batch_with_scaling(tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process audio batch with auto-scaling."""
    auto_scaler = get_global_auto_scaler()
    
    # Process all tasks concurrently
    results = await asyncio.gather(*[
        auto_scaler.process_task(task) for task in tasks
    ])
    
    return results


def create_audio_processing_cluster(min_workers: int = 2, 
                                  max_workers: int = 10,
                                  scaling_policy: str = "hybrid") -> AutoScaler:
    """Create auto-scaling audio processing cluster."""
    policy = ScalingPolicy(scaling_policy.lower())
    
    auto_scaler = AutoScaler(
        min_workers=min_workers,
        max_workers=max_workers,
        policy=policy,
        target_utilization=75.0
    )
    
    # Initialize minimum workers
    for i in range(min_workers):
        worker_id = f"initial_worker_{i}"
        auto_scaler.add_worker(worker_id, capacity=8)
    
    return auto_scaler