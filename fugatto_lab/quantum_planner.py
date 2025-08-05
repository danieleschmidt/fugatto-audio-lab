"""Quantum-Inspired Task Planner for Fugatto Audio Lab.

This module implements quantum-inspired algorithms for intelligent task planning,
resource optimization, and adaptive audio processing workflows.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Task priority levels using quantum-inspired states."""
    CRITICAL = "critical"  # |0⟩ state - immediate execution
    HIGH = "high"         # |1⟩ state - high priority
    MEDIUM = "medium"     # |+⟩ state - balanced superposition
    LOW = "low"          # |-⟩ state - low priority
    DEFERRED = "deferred" # entangled state - context dependent


@dataclass
class QuantumTask:
    """Quantum-inspired task representation with superposition states."""
    
    id: str
    name: str
    description: str
    priority: TaskPriority
    estimated_duration: float  # in seconds
    dependencies: List[str] = field(default_factory=list)
    resources_required: Dict[str, Any] = field(default_factory=dict)
    quantum_state: Dict[str, float] = field(default_factory=dict)
    entangled_tasks: List[str] = field(default_factory=list)
    completion_probability: float = 0.0
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    context: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize quantum state after creation."""
        if not self.quantum_state:
            self.quantum_state = self._initialize_quantum_state()
    
    def _initialize_quantum_state(self) -> Dict[str, float]:
        """Initialize quantum superposition state based on task properties."""
        # Create quantum state vector representing task readiness
        base_state = {
            "ready": 0.7,      # |ready⟩ - task can be executed
            "waiting": 0.2,    # |waiting⟩ - waiting for dependencies
            "blocked": 0.1,    # |blocked⟩ - blocked by resources
            "completed": 0.0   # |completed⟩ - task finished
        }
        
        # Adjust based on dependencies
        if self.dependencies:
            base_state["waiting"] += 0.3
            base_state["ready"] -= 0.3
        
        # Normalize to ensure quantum state coherence
        total = sum(base_state.values())
        return {k: v / total for k, v in base_state.items()}
    
    def collapse_state(self) -> str:
        """Collapse quantum superposition to definite state."""
        # Quantum measurement - collapse to most probable state
        max_prob = max(self.quantum_state.values())
        for state, prob in self.quantum_state.items():
            if prob == max_prob:
                return state
        return "ready"  # fallback
    
    def update_quantum_state(self, new_probabilities: Dict[str, float]):
        """Update quantum state with new probability amplitudes."""
        total = sum(new_probabilities.values())
        if total > 0:
            self.quantum_state = {k: v / total for k, v in new_probabilities.items()}
    
    @property
    def is_ready(self) -> bool:
        """Check if task is ready for execution."""
        return self.quantum_state.get("ready", 0) > 0.5
    
    @property
    def is_completed(self) -> bool:
        """Check if task is completed."""
        return self.quantum_state.get("completed", 0) > 0.9


class QuantumResourceManager:
    """Quantum-inspired resource management with entanglement."""
    
    def __init__(self):
        self.resources = {
            "cpu_cores": 4,
            "memory_gb": 8,
            "gpu_memory_gb": 8,
            "disk_space_gb": 100,
            "network_bandwidth_mbps": 100
        }
        self.allocated_resources = {}
        self.resource_entanglements = {}  # Track resource dependencies
        
    def allocate_resources(self, task_id: str, requirements: Dict[str, Any]) -> bool:
        """Allocate resources using quantum superposition principles."""
        # Check if resources can be allocated
        for resource, amount in requirements.items():
            if resource in self.resources:
                available = self.resources[resource] - sum(
                    self.allocated_resources.get(tid, {}).get(resource, 0)
                    for tid in self.allocated_resources
                )
                if available < amount:
                    logger.warning(f"Insufficient {resource}: need {amount}, have {available}")
                    return False
        
        # Allocate resources with quantum entanglement tracking
        self.allocated_resources[task_id] = requirements.copy()
        
        # Create entanglements for shared resources
        for other_task_id in self.allocated_resources:
            if other_task_id != task_id:
                shared_resources = set(requirements.keys()) & set(
                    self.allocated_resources[other_task_id].keys()
                )
                if shared_resources:
                    self.resource_entanglements[task_id] = other_task_id
        
        logger.debug(f"Allocated resources for task {task_id}: {requirements}")
        return True
    
    def deallocate_resources(self, task_id: str):
        """Free resources and break entanglements."""
        if task_id in self.allocated_resources:
            del self.allocated_resources[task_id]
        
        # Clean up entanglements
        if task_id in self.resource_entanglements:
            del self.resource_entanglements[task_id]
        
        # Remove reverse entanglements
        to_remove = [k for k, v in self.resource_entanglements.items() if v == task_id]
        for k in to_remove:
            del self.resource_entanglements[k]
    
    def get_resource_utilization(self) -> Dict[str, float]:
        """Get current resource utilization percentages."""
        utilization = {}
        for resource, total in self.resources.items():
            used = sum(
                allocation.get(resource, 0)
                for allocation in self.allocated_resources.values()
            )
            utilization[resource] = (used / total) * 100 if total > 0 else 0
        return utilization


class QuantumTaskPlanner:
    """Main quantum-inspired task planner with adaptive optimization."""
    
    def __init__(self, max_concurrent_tasks: int = 4):
        self.tasks = {}  # task_id -> QuantumTask
        self.task_queue = []  # Priority queue with quantum ordering
        self.running_tasks = {}  # task_id -> asyncio.Task
        self.completed_tasks = []
        self.failed_tasks = []
        
        self.resource_manager = QuantumResourceManager()
        self.max_concurrent_tasks = max_concurrent_tasks
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_tasks)
        
        # Quantum-inspired optimization parameters
        self.learning_rate = 0.1
        self.entropy_threshold = 0.5
        self.coherence_time = 300  # 5 minutes
        
        # Performance metrics
        self.metrics = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "average_completion_time": 0.0,
            "resource_efficiency": 0.0,
            "quantum_coherence": 1.0
        }
        
        logger.info(f"QuantumTaskPlanner initialized with {max_concurrent_tasks} concurrent tasks")
    
    def add_task(self, task: QuantumTask) -> str:
        """Add task to quantum planning system."""
        self.tasks[task.id] = task
        self._update_task_queue()
        
        logger.info(f"Added quantum task: {task.name} (Priority: {task.priority.value})")
        return task.id
    
    def create_audio_processing_task(self, 
                                   name: str,
                                   audio_file: str,
                                   operation: str,
                                   parameters: Dict[str, Any],
                                   priority: TaskPriority = TaskPriority.MEDIUM) -> QuantumTask:
        """Create quantum task for audio processing operations."""
        task_id = f"audio_{int(time.time() * 1000)}"
        
        # Estimate duration based on operation complexity
        duration_map = {
            "generate": 30.0,
            "transform": 15.0,
            "analyze": 10.0,
            "convert": 5.0,
            "enhance": 20.0
        }
        estimated_duration = duration_map.get(operation, 10.0)
        
        # Define resource requirements
        resources = {
            "cpu_cores": 1,
            "memory_gb": 2,
            "gpu_memory_gb": 2 if operation in ["generate", "transform"] else 0
        }
        
        task = QuantumTask(
            id=task_id,
            name=name,
            description=f"Audio {operation} operation on {audio_file}",
            priority=priority,
            estimated_duration=estimated_duration,
            resources_required=resources,
            context={
                "audio_file": audio_file,
                "operation": operation,
                "parameters": parameters
            }
        )
        
        return task
    
    def create_batch_processing_task(self,
                                   files: List[str],
                                   operation: str,
                                   batch_size: int = 4) -> List[QuantumTask]:
        """Create quantum tasks for batch audio processing with intelligent chunking."""
        tasks = []
        
        # Create entangled task groups for efficient processing
        chunks = [files[i:i + batch_size] for i in range(0, len(files), batch_size)]
        
        for i, chunk in enumerate(chunks):
            task_id = f"batch_{operation}_{i}_{int(time.time() * 1000)}"
            
            task = QuantumTask(
                id=task_id,
                name=f"Batch {operation} - Chunk {i+1}",
                description=f"Process {len(chunk)} files: {operation}",
                priority=TaskPriority.HIGH,
                estimated_duration=len(chunk) * 15.0,
                resources_required={
                    "cpu_cores": min(2, len(chunk)),
                    "memory_gb": len(chunk) * 1.5,
                    "gpu_memory_gb": 4 if operation in ["generate", "transform"] else 0
                },
                context={
                    "files": chunk,
                    "operation": operation,
                    "batch_index": i,
                    "total_batches": len(chunks)
                }
            )
            
            # Create entanglements between batch tasks
            if i > 0:
                previous_task_id = f"batch_{operation}_{i-1}_{int(time.time() * 1000)}"
                task.entangled_tasks.append(previous_task_id)
            
            tasks.append(task)
        
        return tasks
    
    def optimize_task_order(self) -> List[QuantumTask]:
        """Use quantum-inspired algorithms to optimize task execution order."""
        if not self.task_queue:
            return []
        
        # Quantum annealing-inspired optimization
        current_order = self.task_queue.copy()
        best_order = current_order.copy()
        best_score = self._calculate_execution_score(current_order)
        
        # Simulated quantum annealing
        temperature = 1.0
        cooling_rate = 0.95
        min_temperature = 0.01
        
        while temperature > min_temperature:
            # Quantum tunneling - random perturbations
            if np.random.random() < 0.3:  # Quantum tunneling probability
                # Swap two random tasks
                if len(current_order) >= 2:
                    i, j = np.random.choice(len(current_order), 2, replace=False)
                    current_order[i], current_order[j] = current_order[j], current_order[i]
            
            # Evaluate new configuration
            current_score = self._calculate_execution_score(current_order)
            
            # Accept/reject using quantum probability
            delta = current_score - best_score
            if delta > 0 or np.random.random() < np.exp(delta / temperature):
                best_order = current_order.copy()
                best_score = current_score
            
            temperature *= cooling_rate
        
        logger.debug(f"Optimized task order with score: {best_score:.3f}")
        return best_order
    
    def _calculate_execution_score(self, task_order: List[QuantumTask]) -> float:
        """Calculate execution efficiency score for task order."""
        if not task_order:
            return 0.0
        
        score = 0.0
        
        # Priority scoring
        priority_weights = {
            TaskPriority.CRITICAL: 10.0,
            TaskPriority.HIGH: 5.0,
            TaskPriority.MEDIUM: 2.0,
            TaskPriority.LOW: 1.0,
            TaskPriority.DEFERRED: 0.5
        }
        
        for i, task in enumerate(task_order):
            # Earlier execution of high-priority tasks gets higher score
            position_weight = 1.0 / (i + 1)
            priority_weight = priority_weights[task.priority]
            score += position_weight * priority_weight
            
            # Penalize dependency violations
            for dep_id in task.dependencies:
                dep_positions = [j for j, t in enumerate(task_order) if t.id == dep_id]
                if dep_positions and dep_positions[0] > i:
                    score -= 5.0  # Heavy penalty for dependency violations
            
            # Reward resource efficiency
            quantum_readiness = task.quantum_state.get("ready", 0)
            score += quantum_readiness * 2.0
        
        return score
    
    def _update_task_queue(self):
        """Update quantum task queue with superposition ordering."""
        # Get all non-completed tasks
        pending_tasks = [
            task for task in self.tasks.values()
            if not task.is_completed and task.id not in self.running_tasks
        ]
        
        # Sort by quantum state and priority
        def quantum_sort_key(task: QuantumTask) -> Tuple[float, int, float]:
            readiness = task.quantum_state.get("ready", 0)
            priority_value = {
                TaskPriority.CRITICAL: 4,
                TaskPriority.HIGH: 3,
                TaskPriority.MEDIUM: 2,
                TaskPriority.LOW: 1,
                TaskPriority.DEFERRED: 0
            }[task.priority]
            
            # Calculate quantum advantage score
            quantum_advantage = readiness * priority_value
            
            return (-quantum_advantage, -priority_value, task.estimated_duration)
        
        self.task_queue = sorted(pending_tasks, key=quantum_sort_key)
        
        # Apply quantum optimization
        self.task_queue = self.optimize_task_order()
    
    async def execute_tasks(self) -> Dict[str, Any]:
        """Execute tasks using quantum-inspired scheduling."""
        logger.info("Starting quantum task execution")
        execution_start = time.time()
        
        while (self.task_queue or self.running_tasks) and len(self.completed_tasks + self.failed_tasks) < len(self.tasks):
            # Update quantum states
            self._update_quantum_states()
            
            # Start new tasks if resources available
            while (len(self.running_tasks) < self.max_concurrent_tasks and 
                   self.task_queue):
                
                next_task = self.task_queue[0]
                
                # Check if task is ready (dependencies satisfied)
                if self._check_dependencies(next_task):
                    # Try to allocate resources
                    if self.resource_manager.allocate_resources(
                        next_task.id, next_task.resources_required
                    ):
                        # Start task execution
                        self.task_queue.pop(0)
                        await self._start_task_execution(next_task)
                    else:
                        logger.debug(f"Waiting for resources for task: {next_task.name}")
                        break
                else:
                    logger.debug(f"Dependencies not satisfied for task: {next_task.name}")
                    break
            
            # Check for completed tasks
            await self._check_completed_tasks()
            
            # Quantum coherence maintenance
            await self._maintain_quantum_coherence()
            
            # Short sleep to prevent busy waiting
            await asyncio.sleep(0.1)
        
        execution_time = time.time() - execution_start
        
        # Update metrics
        self._update_metrics(execution_time)
        
        logger.info(f"Quantum task execution completed in {execution_time:.2f}s")
        
        return {
            "execution_time": execution_time,
            "completed_tasks": len(self.completed_tasks),
            "failed_tasks": len(self.failed_tasks),
            "metrics": self.metrics.copy()
        }
    
    def _update_quantum_states(self):
        """Update quantum states of all tasks based on system state."""
        current_time = time.time()
        
        for task in self.tasks.values():
            # Age-based state evolution
            age = current_time - task.created_at
            urgency_factor = min(age / 3600, 1.0)  # Increase urgency over 1 hour
            
            # Update quantum state probabilities
            new_state = task.quantum_state.copy()
            
            # Increase readiness for urgent tasks
            if urgency_factor > 0.5:
                new_state["ready"] = min(new_state["ready"] + urgency_factor * 0.1, 1.0)
            
            # Update based on dependencies
            if self._check_dependencies(task):
                new_state["waiting"] = max(new_state["waiting"] - 0.2, 0.0)
                new_state["ready"] = min(new_state["ready"] + 0.2, 1.0)
            
            # Check resource availability
            if not self.resource_manager.allocate_resources(
                f"check_{task.id}", task.resources_required
            ):
                new_state["blocked"] = min(new_state["blocked"] + 0.1, 1.0)
                new_state["ready"] = max(new_state["ready"] - 0.1, 0.0)
            else:
                # Deallocate the check allocation
                self.resource_manager.deallocate_resources(f"check_{task.id}")
            
            task.update_quantum_state(new_state)
    
    def _check_dependencies(self, task: QuantumTask) -> bool:
        """Check if all task dependencies are satisfied."""
        for dep_id in task.dependencies:
            if dep_id not in [t.id for t in self.completed_tasks]:
                return False
        return True
    
    async def _start_task_execution(self, task: QuantumTask):
        """Start executing a quantum task."""
        task.started_at = time.time()
        
        # Create async task for execution
        async_task = asyncio.create_task(self._execute_single_task(task))
        self.running_tasks[task.id] = async_task
        
        logger.info(f"Started execution of task: {task.name}")
    
    async def _execute_single_task(self, task: QuantumTask):
        """Execute a single quantum task."""
        try:
            # Collapse quantum state to definite execution state
            execution_state = task.collapse_state()
            logger.debug(f"Task {task.name} collapsed to state: {execution_state}")
            
            # Simulate task execution based on context
            if "operation" in task.context:
                await self._execute_audio_operation(task)
            else:
                # Generic task execution
                await asyncio.sleep(task.estimated_duration)
            
            # Mark as completed
            task.completed_at = time.time()
            task.update_quantum_state({"completed": 1.0, "ready": 0.0, "waiting": 0.0, "blocked": 0.0})
            
            self.completed_tasks.append(task)
            logger.info(f"Completed task: {task.name} in {task.completed_at - task.started_at:.2f}s")
            
        except Exception as e:
            logger.error(f"Task {task.name} failed: {e}")
            self.failed_tasks.append(task)
            task.update_quantum_state({"failed": 1.0, "ready": 0.0, "waiting": 0.0, "blocked": 0.0})
    
    async def _execute_audio_operation(self, task: QuantumTask):
        """Execute audio processing operation."""
        operation = task.context["operation"]
        
        # Simulate different audio operations
        if operation == "generate":
            await asyncio.sleep(np.random.uniform(20, 40))  # 20-40 seconds
        elif operation == "transform":
            await asyncio.sleep(np.random.uniform(10, 20))  # 10-20 seconds
        elif operation == "analyze":
            await asyncio.sleep(np.random.uniform(5, 15))   # 5-15 seconds
        elif operation == "convert":
            await asyncio.sleep(np.random.uniform(2, 8))    # 2-8 seconds
        elif operation == "enhance":
            await asyncio.sleep(np.random.uniform(15, 30))  # 15-30 seconds
        else:
            await asyncio.sleep(task.estimated_duration)
        
        # Add some quantum uncertainty
        uncertainty = np.random.uniform(0.8, 1.2)
        await asyncio.sleep(task.estimated_duration * uncertainty * 0.1)
    
    async def _check_completed_tasks(self):
        """Check for completed running tasks and clean up."""
        completed_task_ids = []
        
        for task_id, async_task in self.running_tasks.items():
            if async_task.done():
                try:
                    await async_task  # Get result or exception
                except Exception as e:
                    logger.error(f"Async task {task_id} raised exception: {e}")
                
                completed_task_ids.append(task_id)
                
                # Deallocate resources
                self.resource_manager.deallocate_resources(task_id)
        
        # Remove completed tasks from running tasks
        for task_id in completed_task_ids:
            del self.running_tasks[task_id]
        
        # Update task queue
        if completed_task_ids:
            self._update_task_queue()
    
    async def _maintain_quantum_coherence(self):
        """Maintain quantum coherence of the planning system."""
        # Calculate system coherence
        coherence = self._calculate_system_coherence()
        
        if coherence < self.entropy_threshold:
            logger.debug("Quantum decoherence detected, applying correction")
            
            # Apply decoherence correction
            await self._apply_decoherence_correction()
        
        self.metrics["quantum_coherence"] = coherence
    
    def _calculate_system_coherence(self) -> float:
        """Calculate quantum coherence of the entire system."""
        if not self.tasks:
            return 1.0
        
        total_coherence = 0.0
        for task in self.tasks.values():
            # Calculate individual task coherence
            state_amplitudes = list(task.quantum_state.values())
            # von Neumann entropy as coherence measure
            entropy = -sum(p * np.log2(p + 1e-10) for p in state_amplitudes if p > 0)
            max_entropy = np.log2(len(state_amplitudes))
            coherence = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 1.0
            total_coherence += coherence
        
        return total_coherence / len(self.tasks)
    
    async def _apply_decoherence_correction(self):
        """Apply quantum error correction to maintain system coherence."""
        # Renormalize quantum states
        for task in self.tasks.values():
            total = sum(task.quantum_state.values())
            if total > 0:
                task.quantum_state = {k: v / total for k, v in task.quantum_state.items()}
        
        # Re-entangle related tasks
        for task in self.tasks.values():
            for entangled_id in task.entangled_tasks:
                if entangled_id in self.tasks:
                    entangled_task = self.tasks[entangled_id]
                    # Share quantum state information
                    shared_readiness = (task.quantum_state.get("ready", 0) + 
                                      entangled_task.quantum_state.get("ready", 0)) / 2
                    
                    task.quantum_state["ready"] = shared_readiness
                    entangled_task.quantum_state["ready"] = shared_readiness
    
    def _update_metrics(self, execution_time: float):
        """Update performance metrics."""
        total_tasks = len(self.completed_tasks) + len(self.failed_tasks)
        
        if total_tasks > 0:
            self.metrics["tasks_completed"] = len(self.completed_tasks)
            self.metrics["tasks_failed"] = len(self.failed_tasks)
            
            # Calculate average completion time
            if self.completed_tasks:
                completion_times = [
                    task.completed_at - task.started_at
                    for task in self.completed_tasks
                    if task.started_at and task.completed_at
                ]
                if completion_times:
                    self.metrics["average_completion_time"] = sum(completion_times) / len(completion_times)
            
            # Calculate resource efficiency
            utilization = self.resource_manager.get_resource_utilization()
            avg_utilization = sum(utilization.values()) / len(utilization) if utilization else 0
            self.metrics["resource_efficiency"] = avg_utilization / 100.0
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            "total_tasks": len(self.tasks),
            "pending_tasks": len(self.task_queue),
            "running_tasks": len(self.running_tasks),
            "completed_tasks": len(self.completed_tasks),
            "failed_tasks": len(self.failed_tasks),
            "resource_utilization": self.resource_manager.get_resource_utilization(),
            "quantum_coherence": self.metrics["quantum_coherence"],
            "average_completion_time": self.metrics["average_completion_time"],
            "resource_efficiency": self.metrics["resource_efficiency"]
        }
    
    def save_execution_report(self, filepath: str):
        """Save detailed execution report."""
        report = {
            "execution_summary": self.get_system_status(),
            "completed_tasks": [
                {
                    "id": task.id,
                    "name": task.name,
                    "priority": task.priority.value,
                    "duration": task.completed_at - task.started_at if task.started_at and task.completed_at else None,
                    "resources_used": task.resources_required
                }
                for task in self.completed_tasks
            ],
            "failed_tasks": [
                {
                    "id": task.id,
                    "name": task.name,
                    "priority": task.priority.value,
                    "context": task.context
                }
                for task in self.failed_tasks
            ],
            "metrics": self.metrics,
            "timestamp": time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Execution report saved to: {filepath}")


# Convenience functions for common use cases

def create_audio_generation_pipeline(prompts: List[str], 
                                   priority: TaskPriority = TaskPriority.HIGH) -> QuantumTaskPlanner:
    """Create quantum task planner for audio generation pipeline."""
    planner = QuantumTaskPlanner(max_concurrent_tasks=2)  # GPU memory limited
    
    for i, prompt in enumerate(prompts):
        task = planner.create_audio_processing_task(
            name=f"Generate Audio {i+1}",
            audio_file=f"generated_{i+1}.wav",
            operation="generate",
            parameters={"prompt": prompt, "duration": 10.0},
            priority=priority
        )
        planner.add_task(task)
    
    return planner


def create_batch_enhancement_pipeline(audio_files: List[str]) -> QuantumTaskPlanner:
    """Create quantum task planner for batch audio enhancement."""
    planner = QuantumTaskPlanner(max_concurrent_tasks=4)
    
    # Create batch tasks
    tasks = planner.create_batch_processing_task(
        files=audio_files,
        operation="enhance",
        batch_size=4
    )
    
    for task in tasks:
        planner.add_task(task)
    
    return planner


async def run_quantum_audio_pipeline(planner: QuantumTaskPlanner) -> Dict[str, Any]:
    """Execute quantum audio processing pipeline."""
    logger.info("Starting quantum audio processing pipeline")
    
    try:
        results = await planner.execute_tasks()
        
        # Save execution report
        report_path = f"quantum_execution_report_{int(time.time())}.json"
        planner.save_execution_report(report_path)
        
        return results
    
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise
    
    finally:
        # Cleanup
        planner.executor.shutdown(wait=True)