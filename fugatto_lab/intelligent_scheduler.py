"""Intelligent Scheduler for Fugatto Audio Lab.

Advanced scheduling system with machine learning-based optimization,
adaptive resource management, and predictive task planning.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import time
import json
from pathlib import Path
import pickle
from collections import deque, defaultdict
import threading

logger = logging.getLogger(__name__)


class SchedulingStrategy(Enum):
    """Available scheduling strategies."""
    FIFO = "fifo"                    # First In, First Out
    PRIORITY = "priority"            # Priority-based scheduling
    SJF = "shortest_job_first"       # Shortest Job First
    MULTILEVEL = "multilevel"        # Multi-level feedback queue
    ADAPTIVE = "adaptive"            # Adaptive ML-based scheduling
    QUANTUM_INSPIRED = "quantum"     # Quantum-inspired optimization


@dataclass
class SchedulingMetrics:
    """Comprehensive scheduling performance metrics."""
    
    total_tasks_scheduled: int = 0
    total_execution_time: float = 0.0
    average_waiting_time: float = 0.0
    average_turnaround_time: float = 0.0
    average_response_time: float = 0.0
    throughput: float = 0.0  # tasks per second
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    gpu_utilization: float = 0.0
    context_switches: int = 0
    preemptions: int = 0
    deadline_misses: int = 0
    energy_consumption: float = 0.0
    resource_efficiency: float = 0.0
    fairness_index: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }


@dataclass
class TaskProfile:
    """Detailed task execution profile for learning."""
    
    task_id: str
    task_type: str
    input_size: int
    complexity_score: float
    resource_requirements: Dict[str, float]
    estimated_duration: float
    actual_duration: Optional[float] = None
    waiting_time: float = 0.0
    response_time: float = 0.0
    turnaround_time: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    
    def get_feature_vector(self) -> np.ndarray:
        """Extract feature vector for ML models."""
        features = [
            self.input_size,
            self.complexity_score,
            self.resource_requirements.get('cpu', 0),
            self.resource_requirements.get('memory', 0),
            self.resource_requirements.get('gpu', 0),
            self.estimated_duration,
            time.time() - self.timestamp  # age
        ]
        return np.array(features, dtype=np.float32)


class AdaptiveLearningEngine:
    """Machine learning engine for scheduling optimization."""
    
    def __init__(self):
        self.task_history = deque(maxlen=10000)  # Keep last 10k tasks
        self.performance_history = deque(maxlen=1000)  # Performance samples
        
        # Simple ML models (in production, use scikit-learn or similar)
        self.duration_predictor = None
        self.resource_predictor = None
        self.priority_optimizer = None
        
        # Model parameters
        self.learning_rate = 0.01
        self.model_weights = {}
        self.prediction_accuracy = {"duration": 0.5, "resources": 0.5}
        
        logger.info("AdaptiveLearningEngine initialized")
    
    def add_task_profile(self, profile: TaskProfile):
        """Add completed task profile to learning dataset."""
        self.task_history.append(profile)
        
        # Update model weights based on prediction accuracy
        if profile.actual_duration is not None:
            self._update_duration_model(profile)
        
        # Retrain models periodically
        if len(self.task_history) % 100 == 0:
            self._retrain_models()
    
    def predict_task_duration(self, task_profile: TaskProfile) -> float:
        """Predict task execution duration using ML."""
        if not self.task_history:
            return task_profile.estimated_duration
        
        # Find similar tasks
        similar_tasks = self._find_similar_tasks(task_profile, k=10)
        
        if not similar_tasks:
            return task_profile.estimated_duration
        
        # Weighted average based on similarity
        weights = []
        durations = []
        
        for similar_task, similarity in similar_tasks:
            if similar_task.actual_duration is not None:
                weights.append(similarity)
                durations.append(similar_task.actual_duration)
        
        if not weights:
            return task_profile.estimated_duration
        
        # Calculate weighted prediction
        weights = np.array(weights)
        durations = np.array(durations)
        prediction = np.average(durations, weights=weights)
        
        # Blend with original estimate
        blend_factor = min(self.prediction_accuracy["duration"], 0.8)
        final_prediction = (blend_factor * prediction + 
                          (1 - blend_factor) * task_profile.estimated_duration)
        
        return max(final_prediction, 0.1)  # Minimum 0.1 seconds
    
    def predict_optimal_priority(self, task_profile: TaskProfile, 
                               system_state: Dict[str, Any]) -> float:
        """Predict optimal task priority based on system state."""
        base_priority = 0.5
        
        # Adjust based on resource availability
        cpu_usage = system_state.get('cpu_utilization', 0.5)
        memory_usage = system_state.get('memory_utilization', 0.5)
        queue_length = system_state.get('queue_length', 0)
        
        # Higher priority for low-resource tasks when system is busy
        if cpu_usage > 0.8:
            cpu_requirement = task_profile.resource_requirements.get('cpu', 1.0)
            if cpu_requirement < 0.5:
                base_priority += 0.2
        
        # Adjust for queue length
        if queue_length > 10:
            base_priority -= 0.1 * min(queue_length / 50, 0.3)
        
        # Age-based priority boost
        age = time.time() - task_profile.timestamp
        age_boost = min(age / 3600, 0.3)  # Up to 0.3 boost over 1 hour
        
        return np.clip(base_priority + age_boost, 0.0, 1.0)
    
    def _find_similar_tasks(self, target: TaskProfile, k: int = 5) -> List[Tuple[TaskProfile, float]]:
        """Find k most similar tasks from history."""
        if not self.task_history:
            return []
        
        target_features = target.get_feature_vector()
        similarities = []
        
        for historical_task in self.task_history:
            if historical_task.task_type == target.task_type:
                hist_features = historical_task.get_feature_vector()
                
                # Calculate cosine similarity
                similarity = self._cosine_similarity(target_features, hist_features)
                similarities.append((historical_task, similarity))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def _update_duration_model(self, profile: TaskProfile):
        """Update duration prediction model with new data."""
        if profile.actual_duration is None:
            return
        
        # Calculate prediction error
        predicted = profile.estimated_duration
        actual = profile.actual_duration
        error = abs(predicted - actual) / max(actual, 0.1)
        
        # Update accuracy metric
        self.prediction_accuracy["duration"] = (
            0.9 * self.prediction_accuracy["duration"] + 
            0.1 * (1.0 - min(error, 1.0))
        )
    
    def _retrain_models(self):
        """Retrain ML models with accumulated data."""
        if len(self.task_history) < 50:
            return
        
        logger.debug(f"Retraining models with {len(self.task_history)} samples")
        
        # Simple retraining - in production use proper ML frameworks
        recent_tasks = list(self.task_history)[-500:]  # Use recent 500 tasks
        
        # Update prediction accuracy based on recent performance
        duration_errors = []
        for task in recent_tasks:
            if task.actual_duration is not None:
                error = abs(task.estimated_duration - task.actual_duration) / max(task.actual_duration, 0.1)
                duration_errors.append(error)
        
        if duration_errors:
            avg_error = np.mean(duration_errors)
            self.prediction_accuracy["duration"] = max(0.1, 1.0 - avg_error)
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get learning engine statistics."""
        return {
            "task_history_size": len(self.task_history),
            "performance_history_size": len(self.performance_history),
            "duration_prediction_accuracy": self.prediction_accuracy["duration"],
            "resource_prediction_accuracy": self.prediction_accuracy["resources"]
        }


class IntelligentScheduler:
    """Advanced intelligent scheduler with ML optimization."""
    
    def __init__(self, 
                 strategy: SchedulingStrategy = SchedulingStrategy.ADAPTIVE,
                 max_concurrent_tasks: int = 4,
                 enable_learning: bool = True):
        
        self.strategy = strategy
        self.max_concurrent_tasks = max_concurrent_tasks
        self.enable_learning = enable_learning
        
        # Task queues for different priority levels
        self.high_priority_queue = deque()
        self.medium_priority_queue = deque()
        self.low_priority_queue = deque()
        self.running_tasks = {}
        
        # Scheduling state
        self.completed_tasks = []
        self.failed_tasks = []
        self.task_profiles = {}
        
        # Learning and optimization
        self.learning_engine = AdaptiveLearningEngine() if enable_learning else None
        self.metrics = SchedulingMetrics()
        
        # Resource monitoring
        self.resource_monitor = ResourceMonitor()
        self.system_state = {}
        
        # Scheduling parameters
        self.time_quantum = 1.0  # For round-robin
        self.aging_factor = 0.1   # Priority aging
        self.preemption_enabled = False
        
        # Performance tracking
        self.scheduling_overhead = 0.0
        self.last_schedule_time = time.time()
        
        logger.info(f"IntelligentScheduler initialized: {strategy.value}, max_tasks={max_concurrent_tasks}")
    
    def schedule_task(self, task_id: str, task_info: Dict[str, Any], 
                     priority: float = 0.5) -> bool:
        """Schedule a new task using intelligent algorithms."""
        scheduling_start = time.time()
        
        # Create task profile
        profile = TaskProfile(
            task_id=task_id,
            task_type=task_info.get('type', 'generic'),
            input_size=task_info.get('input_size', 1000),
            complexity_score=task_info.get('complexity', 1.0),
            resource_requirements=task_info.get('resources', {}),
            estimated_duration=task_info.get('duration', 10.0)
        )
        
        # Use ML to refine estimates
        if self.learning_engine:
            # Predict more accurate duration
            predicted_duration = self.learning_engine.predict_task_duration(profile)
            profile.estimated_duration = predicted_duration
            
            # Optimize priority based on system state
            optimal_priority = self.learning_engine.predict_optimal_priority(
                profile, self.system_state
            )
            priority = optimal_priority
        
        self.task_profiles[task_id] = profile
        
        # Add to appropriate queue based on strategy
        scheduled = self._add_to_queue(task_id, profile, priority)
        
        if scheduled:
            self.metrics.total_tasks_scheduled += 1
            
            # Trigger scheduling decision
            asyncio.create_task(self._schedule_next_tasks())
        
        # Update scheduling overhead
        scheduling_time = time.time() - scheduling_start
        self.scheduling_overhead = (0.9 * self.scheduling_overhead + 
                                  0.1 * scheduling_time)
        
        return scheduled
    
    def _add_to_queue(self, task_id: str, profile: TaskProfile, priority: float) -> bool:
        """Add task to appropriate queue based on priority and strategy."""
        if self.strategy == SchedulingStrategy.FIFO:
            self.medium_priority_queue.append((task_id, profile, priority))
            
        elif self.strategy == SchedulingStrategy.PRIORITY:
            if priority > 0.7:
                self.high_priority_queue.append((task_id, profile, priority))
            elif priority > 0.3:
                self.medium_priority_queue.append((task_id, profile, priority))
            else:
                self.low_priority_queue.append((task_id, profile, priority))
                
        elif self.strategy == SchedulingStrategy.SJF:
            # Insert in order of estimated duration
            queue = self.medium_priority_queue
            inserted = False
            for i, (_, existing_profile, _) in enumerate(queue):
                if profile.estimated_duration < existing_profile.estimated_duration:
                    queue.insert(i, (task_id, profile, priority))
                    inserted = True
                    break
            if not inserted:
                queue.append((task_id, profile, priority))
                
        elif self.strategy == SchedulingStrategy.ADAPTIVE:
            # ML-based queue selection
            self._adaptive_queue_selection(task_id, profile, priority)
            
        else:  # Default to priority-based
            if priority > 0.7:
                self.high_priority_queue.append((task_id, profile, priority))
            else:
                self.medium_priority_queue.append((task_id, profile, priority))
        
        return True
    
    def _adaptive_queue_selection(self, task_id: str, profile: TaskProfile, priority: float):
        """Intelligently select queue based on ML predictions."""
        # Analyze system load
        cpu_usage = self.system_state.get('cpu_utilization', 0.5)
        queue_lengths = {
            'high': len(self.high_priority_queue),
            'medium': len(self.medium_priority_queue),
            'low': len(self.low_priority_queue)
        }
        
        # Decision factors
        factors = {
            'urgency': priority,
            'resource_requirement': sum(profile.resource_requirements.values()),
            'estimated_duration': profile.estimated_duration,
            'queue_balance': 1.0 - (queue_lengths['high'] / max(sum(queue_lengths.values()), 1))
        }
        
        # Calculate adaptive priority
        adaptive_priority = (
            0.4 * factors['urgency'] +
            0.2 * (1.0 - factors['resource_requirement'] / 10.0) +
            0.2 * (1.0 - min(factors['estimated_duration'] / 60.0, 1.0)) +
            0.2 * factors['queue_balance']
        )
        
        # Select queue
        if adaptive_priority > 0.7 or (cpu_usage < 0.5 and priority > 0.6):
            self.high_priority_queue.append((task_id, profile, adaptive_priority))
        elif adaptive_priority > 0.3:
            self.medium_priority_queue.append((task_id, profile, adaptive_priority))
        else:
            self.low_priority_queue.append((task_id, profile, adaptive_priority))
    
    async def _schedule_next_tasks(self):
        """Schedule next tasks for execution."""
        while len(self.running_tasks) < self.max_concurrent_tasks:
            # Update system state
            self.system_state = self.resource_monitor.get_system_state()
            
            # Select next task
            next_task = self._select_next_task()
            if not next_task:
                break
            
            task_id, profile, priority = next_task
            
            # Check resource availability
            if not self._check_resource_availability(profile):
                # Put task back in queue if resources not available
                self._add_to_queue(task_id, profile, priority)
                break
            
            # Start task execution
            await self._start_task_execution(task_id, profile)
    
    def _select_next_task(self) -> Optional[Tuple[str, TaskProfile, float]]:
        """Select next task based on scheduling strategy."""
        if self.strategy == SchedulingStrategy.FIFO:
            return self._select_fifo()
        elif self.strategy == SchedulingStrategy.PRIORITY:
            return self._select_priority()
        elif self.strategy == SchedulingStrategy.SJF:
            return self._select_sjf()
        elif self.strategy == SchedulingStrategy.MULTILEVEL:
            return self._select_multilevel()
        elif self.strategy == SchedulingStrategy.ADAPTIVE:
            return self._select_adaptive()
        else:
            return self._select_priority()  # Default
    
    def _select_fifo(self) -> Optional[Tuple[str, TaskProfile, float]]:
        """First-In-First-Out selection."""
        if self.medium_priority_queue:
            return self.medium_priority_queue.popleft()
        return None
    
    def _select_priority(self) -> Optional[Tuple[str, TaskProfile, float]]:
        """Priority-based selection with aging."""
        # Apply aging to prevent starvation
        self._apply_priority_aging()
        
        # Select from highest priority queue first
        if self.high_priority_queue:
            return self.high_priority_queue.popleft()
        elif self.medium_priority_queue:
            return self.medium_priority_queue.popleft()
        elif self.low_priority_queue:
            return self.low_priority_queue.popleft()
        return None
    
    def _select_sjf(self) -> Optional[Tuple[str, TaskProfile, float]]:
        """Shortest Job First selection."""
        # Queue is already sorted by duration
        if self.medium_priority_queue:
            return self.medium_priority_queue.popleft()
        return None
    
    def _select_multilevel(self) -> Optional[Tuple[str, TaskProfile, float]]:
        """Multi-level feedback queue selection."""
        # Implement time quantum and queue demotion
        current_time = time.time()
        
        # Check for time quantum expiration in running tasks
        for task_id, task_info in list(self.running_tasks.items()):
            runtime = current_time - task_info['start_time']
            if runtime > self.time_quantum:
                # Preempt and demote
                if self.preemption_enabled:
                    self._preempt_task(task_id)
        
        # Select from queues with round-robin at each level
        if self.high_priority_queue:
            return self.high_priority_queue.popleft()
        elif self.medium_priority_queue:
            return self.medium_priority_queue.popleft()
        elif self.low_priority_queue:
            return self.low_priority_queue.popleft()
        return None
    
    def _select_adaptive(self) -> Optional[Tuple[str, TaskProfile, float]]:
        """ML-based adaptive selection."""
        all_tasks = []
        
        # Collect all tasks with queue info
        for task_id, profile, priority in self.high_priority_queue:
            all_tasks.append((task_id, profile, priority, 'high'))
        for task_id, profile, priority in self.medium_priority_queue:
            all_tasks.append((task_id, profile, priority, 'medium'))
        for task_id, profile, priority in self.low_priority_queue:
            all_tasks.append((task_id, profile, priority, 'low'))
        
        if not all_tasks:
            return None
        
        # Use ML to score tasks
        best_task = None
        best_score = -1.0
        
        for task_id, profile, priority, queue_level in all_tasks:
            score = self._calculate_adaptive_score(profile, priority, queue_level)
            if score > best_score:
                best_score = score
                best_task = (task_id, profile, priority, queue_level)
        
        if best_task:
            task_id, profile, priority, queue_level = best_task
            
            # Remove from appropriate queue
            if queue_level == 'high':
                self.high_priority_queue.remove((task_id, profile, priority))
            elif queue_level == 'medium':
                self.medium_priority_queue.remove((task_id, profile, priority))
            else:
                self.low_priority_queue.remove((task_id, profile, priority))
            
            return (task_id, profile, priority)
        
        return None
    
    def _calculate_adaptive_score(self, profile: TaskProfile, priority: float, queue_level: str) -> float:
        """Calculate adaptive score for task selection."""
        base_score = priority
        
        # System load factor
        cpu_usage = self.system_state.get('cpu_utilization', 0.5)
        memory_usage = self.system_state.get('memory_utilization', 0.5)
        
        # Prefer light tasks when system is busy
        if cpu_usage > 0.8:
            resource_factor = 1.0 - (sum(profile.resource_requirements.values()) / 10.0)
            base_score += 0.2 * resource_factor
        
        # Age factor
        age = time.time() - profile.timestamp
        age_factor = min(age / 3600, 0.5)  # Up to 0.5 boost over 1 hour
        base_score += age_factor
        
        # Queue level factor
        queue_factors = {'high': 1.0, 'medium': 0.7, 'low': 0.4}
        base_score *= queue_factors.get(queue_level, 0.5)
        
        # Predicted completion benefit
        if self.learning_engine:
            predicted_duration = self.learning_engine.predict_task_duration(profile)
            duration_factor = 1.0 - min(predicted_duration / 60.0, 0.5)
            base_score += 0.1 * duration_factor
        
        return base_score
    
    def _apply_priority_aging(self):
        """Apply aging to prevent starvation."""
        current_time = time.time()
        
        # Age tasks in medium and low priority queues
        for queue in [self.medium_priority_queue, self.low_priority_queue]:
            for i, (task_id, profile, priority) in enumerate(queue):
                age = current_time - profile.timestamp
                age_boost = min(age / 3600 * self.aging_factor, 0.3)  # Max 0.3 boost
                new_priority = min(priority + age_boost, 1.0)
                queue[i] = (task_id, profile, new_priority)
        
        # Promote aged tasks from low to medium priority
        promoted = []
        for i, (task_id, profile, priority) in enumerate(self.low_priority_queue):
            if priority > 0.6:  # Promotion threshold
                promoted.append(i)
                self.medium_priority_queue.append((task_id, profile, priority))
        
        # Remove promoted tasks (in reverse order to maintain indices)
        for i in reversed(promoted):
            del self.low_priority_queue[i]
    
    def _check_resource_availability(self, profile: TaskProfile) -> bool:
        """Check if required resources are available."""
        system_state = self.resource_monitor.get_system_state()
        
        # Check CPU availability
        cpu_required = profile.resource_requirements.get('cpu', 1.0)
        if system_state['cpu_utilization'] + cpu_required * 25 > 95:
            return False
        
        # Check memory availability
        memory_required = profile.resource_requirements.get('memory', 1.0)
        if system_state['memory_utilization'] + memory_required * 12.5 > 90:
            return False
        
        # Check GPU availability if required
        gpu_required = profile.resource_requirements.get('gpu', 0)
        if gpu_required > 0 and system_state['gpu_utilization'] > 85:
            return False
        
        return True
    
    async def _start_task_execution(self, task_id: str, profile: TaskProfile):
        """Start executing a scheduled task."""
        start_time = time.time()
        profile.response_time = start_time - profile.timestamp
        
        # Create execution context
        task_context = {
            'start_time': start_time,
            'profile': profile,
            'executor': asyncio.create_task(self._execute_task(task_id, profile))
        }
        
        self.running_tasks[task_id] = task_context
        
        logger.info(f"Started task execution: {task_id}")
        
        # Monitor task completion
        asyncio.create_task(self._monitor_task_completion(task_id))
    
    async def _execute_task(self, task_id: str, profile: TaskProfile):
        """Execute the actual task (simulated)."""
        try:
            # Simulate task execution
            execution_time = profile.estimated_duration
            
            # Add some variability
            variability = np.random.uniform(0.8, 1.2)
            actual_time = execution_time * variability
            
            await asyncio.sleep(actual_time)
            
            # Update profile with actual results
            profile.actual_duration = actual_time
            profile.success = True
            
            return {"success": True, "duration": actual_time}
            
        except Exception as e:
            profile.success = False
            profile.error_message = str(e)
            logger.error(f"Task {task_id} failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _monitor_task_completion(self, task_id: str):
        """Monitor task for completion and update metrics."""
        task_context = self.running_tasks[task_id]
        profile = task_context['profile']
        
        try:
            # Wait for task completion
            result = await task_context['executor']
            
            # Update timing metrics
            completion_time = time.time()
            profile.turnaround_time = completion_time - profile.timestamp
            profile.waiting_time = task_context['start_time'] - profile.timestamp
            
            # Add to completed tasks
            self.completed_tasks.append(profile)
            
            # Update learning engine
            if self.learning_engine:
                self.learning_engine.add_task_profile(profile)
            
            # Update metrics
            self._update_metrics(profile)
            
            logger.info(f"Task {task_id} completed successfully in {profile.actual_duration:.2f}s")
            
        except Exception as e:
            profile.success = False
            profile.error_message = str(e)
            self.failed_tasks.append(profile)
            logger.error(f"Task {task_id} monitoring failed: {e}")
        
        finally:
            # Clean up
            if task_id in self.running_tasks:
                del self.running_tasks[task_id]
            
            # Schedule next tasks
            await self._schedule_next_tasks()
    
    def _preempt_task(self, task_id: str):
        """Preempt a running task (for multilevel scheduling)."""
        if task_id in self.running_tasks:
            task_context = self.running_tasks[task_id]
            profile = task_context['profile']
            
            # Cancel the task
            task_context['executor'].cancel()
            
            # Move to lower priority queue
            self.low_priority_queue.append((task_id, profile, 0.3))
            
            # Update metrics
            self.metrics.preemptions += 1
            
            del self.running_tasks[task_id]
            
            logger.info(f"Preempted task: {task_id}")
    
    def _update_metrics(self, profile: TaskProfile):
        """Update scheduling metrics."""
        self.metrics.total_execution_time += profile.actual_duration or 0
        
        # Running averages
        n = len(self.completed_tasks)
        if n > 0:
            self.metrics.average_waiting_time = (
                (self.metrics.average_waiting_time * (n - 1) + profile.waiting_time) / n
            )
            self.metrics.average_turnaround_time = (
                (self.metrics.average_turnaround_time * (n - 1) + profile.turnaround_time) / n
            )
            self.metrics.average_response_time = (
                (self.metrics.average_response_time * (n - 1) + profile.response_time) / n
            )
        
        # Throughput
        if self.metrics.total_execution_time > 0:
            self.metrics.throughput = len(self.completed_tasks) / self.metrics.total_execution_time
        
        # Resource utilization (from system monitor)
        sys_state = self.resource_monitor.get_system_state()
        self.metrics.cpu_utilization = sys_state['cpu_utilization']
        self.metrics.memory_utilization = sys_state['memory_utilization']
        self.metrics.gpu_utilization = sys_state['gpu_utilization']
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """Get comprehensive scheduler status."""
        return {
            "strategy": self.strategy.value,
            "queues": {
                "high_priority": len(self.high_priority_queue),
                "medium_priority": len(self.medium_priority_queue),
                "low_priority": len(self.low_priority_queue)
            },
            "running_tasks": len(self.running_tasks),
            "completed_tasks": len(self.completed_tasks),
            "failed_tasks": len(self.failed_tasks),
            "metrics": self.metrics.to_dict(),
            "system_state": self.system_state,
            "scheduling_overhead": self.scheduling_overhead,
            "learning_stats": self.learning_engine.get_learning_stats() if self.learning_engine else {}
        }
    
    def save_performance_report(self, filepath: str):
        """Save detailed performance report."""
        report = {
            "scheduler_config": {
                "strategy": self.strategy.value,
                "max_concurrent_tasks": self.max_concurrent_tasks,
                "enable_learning": self.enable_learning
            },
            "performance_metrics": self.metrics.to_dict(),
            "task_statistics": {
                "total_scheduled": self.metrics.total_tasks_scheduled,
                "completed": len(self.completed_tasks),
                "failed": len(self.failed_tasks),
                "success_rate": len(self.completed_tasks) / max(self.metrics.total_tasks_scheduled, 1)
            },
            "resource_utilization": self.system_state,
            "learning_performance": self.learning_engine.get_learning_stats() if self.learning_engine else {},
            "timestamp": time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Performance report saved to: {filepath}")


class ResourceMonitor:
    """System resource monitoring for intelligent scheduling."""
    
    def __init__(self, update_interval: float = 1.0):
        self.update_interval = update_interval
        self.last_update = 0
        self.cached_state = {
            'cpu_utilization': 50.0,
            'memory_utilization': 40.0,
            'gpu_utilization': 30.0,
            'disk_io': 20.0,
            'network_io': 10.0,
            'load_average': 1.0,
            'available_memory_gb': 4.0,
            'queue_length': 0
        }
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitor_resources, daemon=True)
        self.monitoring_thread.start()
    
    def get_system_state(self) -> Dict[str, float]:
        """Get current system resource state."""
        current_time = time.time()
        
        if current_time - self.last_update > self.update_interval:
            self._update_system_state()
            self.last_update = current_time
        
        return self.cached_state.copy()
    
    def _monitor_resources(self):
        """Background resource monitoring."""
        while True:
            try:
                self._update_system_state()
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                time.sleep(5.0)
    
    def _update_system_state(self):
        """Update system resource measurements."""
        # Simulate system resource usage
        # In production, use psutil or similar
        
        # CPU utilization with some variability
        base_cpu = 30 + 20 * np.sin(time.time() / 60)  # 60-second cycle
        self.cached_state['cpu_utilization'] = max(0, min(100, base_cpu + np.random.normal(0, 5)))
        
        # Memory utilization
        base_memory = 40 + 10 * np.sin(time.time() / 120)  # 2-minute cycle
        self.cached_state['memory_utilization'] = max(0, min(90, base_memory + np.random.normal(0, 3)))
        
        # GPU utilization (more volatile)
        self.cached_state['gpu_utilization'] = max(0, min(100, 
            30 + np.random.normal(0, 15)))
        
        # I/O metrics
        self.cached_state['disk_io'] = max(0, min(100, 20 + np.random.normal(0, 10)))
        self.cached_state['network_io'] = max(0, min(100, 10 + np.random.normal(0, 5)))
        
        # Load average
        self.cached_state['load_average'] = max(0.1, 1.0 + np.random.normal(0, 0.3))
        
        # Available memory
        memory_usage = self.cached_state['memory_utilization']
        total_memory = 8.0  # GB
        self.cached_state['available_memory_gb'] = total_memory * (1 - memory_usage / 100)


# Convenience functions and factory methods

def create_intelligent_scheduler(strategy: str = "adaptive", 
                                max_tasks: int = 4,
                                enable_ml: bool = True) -> IntelligentScheduler:
    """Create intelligent scheduler with specified configuration."""
    strategy_enum = SchedulingStrategy(strategy.lower())
    return IntelligentScheduler(
        strategy=strategy_enum,
        max_concurrent_tasks=max_tasks,
        enable_learning=enable_ml
    )


async def run_audio_processing_batch(scheduler: IntelligentScheduler,
                                   audio_files: List[str],
                                   operation: str = "enhance") -> Dict[str, Any]:
    """Run batch audio processing with intelligent scheduling."""
    logger.info(f"Starting batch {operation} processing for {len(audio_files)} files")
    
    # Schedule all tasks
    scheduled_tasks = []
    for i, audio_file in enumerate(audio_files):
        task_info = {
            'type': 'audio_processing',
            'input_size': 1000 + i * 100,  # Simulate varying file sizes
            'complexity': 1.0 + (i % 3) * 0.5,  # Varying complexity
            'resources': {
                'cpu': 1.0 + (i % 2) * 0.5,
                'memory': 2.0 + (i % 3) * 1.0,
                'gpu': 1.0 if operation in ['generate', 'transform'] else 0
            },
            'duration': 10.0 + (i % 5) * 5.0  # 10-35 seconds
        }
        
        priority = 0.7 if i < len(audio_files) // 2 else 0.4  # Half high priority
        task_id = f"{operation}_task_{i}"
        
        success = scheduler.schedule_task(task_id, task_info, priority)
        if success:
            scheduled_tasks.append(task_id)
    
    # Wait for all tasks to complete
    start_time = time.time()
    while (len(scheduler.completed_tasks) + len(scheduler.failed_tasks)) < len(scheduled_tasks):
        await asyncio.sleep(1.0)
        
        # Timeout after 10 minutes
        if time.time() - start_time > 600:
            logger.warning("Batch processing timeout")
            break
    
    total_time = time.time() - start_time
    
    # Generate results
    results = {
        "total_tasks": len(scheduled_tasks),
        "completed": len(scheduler.completed_tasks),
        "failed": len(scheduler.failed_tasks),
        "total_time": total_time,
        "scheduler_status": scheduler.get_scheduler_status()
    }
    
    # Save performance report
    report_path = f"scheduler_report_{operation}_{int(time.time())}.json"
    scheduler.save_performance_report(report_path)
    
    logger.info(f"Batch processing completed: {results['completed']}/{results['total_tasks']} successful")
    
    return results