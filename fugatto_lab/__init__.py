"""Fugatto Audio Lab - Advanced Audio Generation with Quantum Task Planning.

Enterprise-grade AI audio platform with quantum-inspired task planning,
intelligent scheduling, and autonomous scaling. Built with advanced 
Generation 1, 2, 3 enhancements for production deployment.
"""

__version__ = "0.3.0"
__author__ = "Daniel Schmidt & Terragon Labs"
__email__ = "daniel@example.com"

# Core quantum planner components (dependency-free)
from fugatto_lab.quantum_planner import (
    QuantumTaskPlanner, 
    QuantumTask, 
    TaskPriority,
    QuantumResourceManager,
    create_audio_generation_pipeline,
    create_batch_enhancement_pipeline,
    run_quantum_audio_pipeline
)

# Enhanced components (conditional imports)
try:
    from fugatto_lab.robust_validation import (
        RobustValidator,
        EnhancedErrorHandler,
        MonitoringEnhancer,
        create_robust_quantum_planner
    )
    HAS_ROBUST_VALIDATION = True
except ImportError:
    HAS_ROBUST_VALIDATION = False

try:
    from fugatto_lab.performance_scaler import (
        AdvancedPerformanceOptimizer,
        AutoScaler,
        BottleneckDetector,
        enhance_planner_with_scaling
    )
    HAS_PERFORMANCE_SCALING = True
except ImportError:
    HAS_PERFORMANCE_SCALING = False

# Optional heavy dependencies
try:
    from fugatto_lab.core import FugattoModel, AudioProcessor
    HAS_AUDIO_CORE = True
except ImportError:
    HAS_AUDIO_CORE = False

try:
    from fugatto_lab.intelligent_scheduler import IntelligentScheduler, SchedulingStrategy
    HAS_INTELLIGENT_SCHEDULER = True
except ImportError:
    HAS_INTELLIGENT_SCHEDULER = False

# Generation 5.0: Quantum Consciousness Components
try:
    from fugatto_lab.quantum_consciousness_monitor import (
        QuantumConsciousnessMonitor,
        ConsciousnessEvent,
        AwarenessType,
        ConsciousnessLevel,
        create_quantum_consciousness_monitor
    )
    HAS_QUANTUM_CONSCIOUSNESS = True
except ImportError:
    HAS_QUANTUM_CONSCIOUSNESS = False

try:
    from fugatto_lab.temporal_consciousness_audio_processor_v5 import (
        TemporalConsciousnessAudioProcessor,
        AudioConsciousnessVector,
        AudioConsciousnessState,
        TemporalAudioSegment,
        create_temporal_consciousness_processor
    )
    HAS_TEMPORAL_CONSCIOUSNESS = True
except ImportError:
    HAS_TEMPORAL_CONSCIOUSNESS = False

# Core exports (always available)
__all__ = [
    "QuantumTaskPlanner",
    "QuantumTask", 
    "TaskPriority",
    "QuantumResourceManager",
    "create_audio_generation_pipeline",
    "create_batch_enhancement_pipeline",
    "run_quantum_audio_pipeline",
    "__version__"
]

# Conditional exports based on available dependencies
if HAS_ROBUST_VALIDATION:
    __all__.extend([
        "RobustValidator",
        "EnhancedErrorHandler", 
        "MonitoringEnhancer",
        "create_robust_quantum_planner"
    ])

if HAS_PERFORMANCE_SCALING:
    __all__.extend([
        "AdvancedPerformanceOptimizer",
        "AutoScaler",
        "BottleneckDetector",
        "enhance_planner_with_scaling"
    ])

if HAS_AUDIO_CORE:
    __all__.extend(["FugattoModel", "AudioProcessor"])

if HAS_INTELLIGENT_SCHEDULER:
    __all__.extend(["IntelligentScheduler", "SchedulingStrategy"])

# Generation 5.0: Quantum Consciousness exports
if HAS_QUANTUM_CONSCIOUSNESS:
    __all__.extend([
        "QuantumConsciousnessMonitor",
        "ConsciousnessEvent", 
        "AwarenessType",
        "ConsciousnessLevel",
        "create_quantum_consciousness_monitor"
    ])

if HAS_TEMPORAL_CONSCIOUSNESS:
    __all__.extend([
        "TemporalConsciousnessAudioProcessor",
        "AudioConsciousnessVector",
        "AudioConsciousnessState", 
        "TemporalAudioSegment",
        "create_temporal_consciousness_processor"
    ])

# Feature flags for runtime checking
FEATURES = {
    "quantum_planning": True,           # Always available
    "robust_validation": HAS_ROBUST_VALIDATION,
    "performance_scaling": HAS_PERFORMANCE_SCALING,
    "audio_core": HAS_AUDIO_CORE,
    "intelligent_scheduling": HAS_INTELLIGENT_SCHEDULER,
    "quantum_consciousness": HAS_QUANTUM_CONSCIOUSNESS,         # Generation 5.0
    "temporal_consciousness": HAS_TEMPORAL_CONSCIOUSNESS        # Generation 5.0
}