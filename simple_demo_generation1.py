#!/usr/bin/env python3
"""Simple Generation 1 Demo: Basic Fugatto Audio Lab Functionality.

This simplified demo showcases the core functionality without complex
mathematical operations that require full numpy compatibility.
"""

import sys
import time
import asyncio
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from fugatto_lab import QuantumTaskPlanner, QuantumTask, TaskPriority


async def demo_basic_quantum_planning():
    """Demonstrate basic quantum task planning capabilities."""
    
    print("⚡ QUANTUM TASK PLANNING DEMO")
    print("=" * 40)
    
    # Initialize quantum planner
    planner = QuantumTaskPlanner()
    print("✅ Quantum task planner initialized")
    
    # Create sample tasks
    tasks = [
        QuantumTask(
            id="audio_gen_1", 
            name="Generate cat meowing",
            priority=TaskPriority.HIGH,
            estimated_duration=3.0,
            resource_requirements={"cpu": 2, "memory": 1}
        ),
        QuantumTask(
            id="audio_gen_2",
            name="Generate ocean waves", 
            priority=TaskPriority.MEDIUM,
            estimated_duration=5.0,
            resource_requirements={"cpu": 1, "memory": 2}
        ),
        QuantumTask(
            id="process_1",
            name="Apply reverb effect",
            priority=TaskPriority.LOW,
            estimated_duration=1.0,
            resource_requirements={"cpu": 1, "memory": 1},
            dependencies=["audio_gen_1"]
        )
    ]
    
    print(f"📋 Created {len(tasks)} quantum tasks")
    
    # Plan execution
    print("\n🔄 Planning quantum task execution...")
    start_time = time.time()
    
    execution_plan = await planner.create_execution_plan(tasks)
    
    planning_time = time.time() - start_time
    print(f"✅ Execution plan created in {planning_time:.3f}s")
    
    # Execute plan
    print(f"📊 Execution order: {[step.task_id for step in execution_plan]}")
    
    return execution_plan


async def demo_basic_imports():
    """Demonstrate that core imports work correctly."""
    
    print("🔧 CORE IMPORTS DEMO")
    print("=" * 30)
    
    # Test quantum planner imports
    try:
        from fugatto_lab import QuantumTaskPlanner, QuantumResourceManager
        print("✅ Quantum planning components imported")
    except ImportError as e:
        print(f"❌ Failed to import quantum components: {e}")
        return False
    
    # Test conditional imports
    try:
        from fugatto_lab import FEATURES
        print("✅ Feature flags loaded:")
        for feature, available in FEATURES.items():
            status = "✅" if available else "❌"
            print(f"    {status} {feature}: {available}")
    except ImportError as e:
        print(f"❌ Failed to import features: {e}")
        return False
    
    # Test core model (with mock dependencies)
    try:
        from fugatto_lab.core import FugattoModel
        model = FugattoModel()
        print(f"✅ Core model initialized: {model.model_name}")
        info = model.get_model_info()
        print(f"    • Device: {info['device']}")
        print(f"    • Sample rate: {info['sample_rate']}Hz")
        print(f"    • Loaded: {info['loaded']}")
    except Exception as e:
        print(f"❌ Core model failed: {e}")
        return False
    
    return True


async def demo_quantum_resource_management():
    """Demonstrate quantum resource management."""
    
    print("\n🎛️ QUANTUM RESOURCE MANAGEMENT DEMO")
    print("=" * 45)
    
    # Initialize resource manager
    from fugatto_lab import QuantumResourceManager
    resource_manager = QuantumResourceManager()
    
    print("✅ Quantum resource manager initialized")
    
    # Check available resources
    resources = resource_manager.get_available_resources()
    print(f"📊 Available resources:")
    for resource, amount in resources.items():
        print(f"    • {resource}: {amount}")
    
    # Simulate resource allocation
    allocation_request = {"cpu": 3, "memory": 4, "gpu": 1}
    print(f"\n🔄 Requesting resources: {allocation_request}")
    
    can_allocate = resource_manager.can_allocate(allocation_request)
    print(f"✅ Allocation possible: {can_allocate}")
    
    if can_allocate:
        allocation_id = resource_manager.allocate_resources(allocation_request)
        print(f"🎯 Resources allocated with ID: {allocation_id}")
        
        # Release resources
        resource_manager.release_resources(allocation_id)
        print("✅ Resources released successfully")
    
    return True


async def main():
    """Main demo function."""
    
    print("🚀 FUGATTO AUDIO LAB SIMPLE DEMO")
    print("Generation 1: Core Functionality Verification")
    print("=" * 60)
    
    success_count = 0
    total_tests = 0
    
    try:
        # Test 1: Core imports
        total_tests += 1
        print(f"\n🧪 TEST {total_tests}: Core Imports")
        if await demo_basic_imports():
            success_count += 1
            print("✅ PASSED")
        else:
            print("❌ FAILED")
        
        # Test 2: Quantum planning
        total_tests += 1
        print(f"\n🧪 TEST {total_tests}: Quantum Task Planning")
        try:
            plan = await demo_basic_quantum_planning()
            if plan and len(plan) > 0:
                success_count += 1
                print("✅ PASSED")
            else:
                print("❌ FAILED: No execution plan generated")
        except Exception as e:
            print(f"❌ FAILED: {e}")
        
        # Test 3: Resource management
        total_tests += 1
        print(f"\n🧪 TEST {total_tests}: Resource Management")
        try:
            if await demo_quantum_resource_management():
                success_count += 1
                print("✅ PASSED")
            else:
                print("❌ FAILED")
        except Exception as e:
            print(f"❌ FAILED: {e}")
        
        # Final results
        print("\n" + "=" * 60)
        print("🏆 DEMO RESULTS")
        print("=" * 60)
        print(f"📊 Tests passed: {success_count}/{total_tests}")
        success_rate = (success_count / total_tests) * 100
        print(f"📈 Success rate: {success_rate:.1f}%")
        
        if success_count == total_tests:
            print("\n🎉 ALL TESTS PASSED!")
            print("✅ Core Fugatto Audio Lab functionality is working")
            print("✅ Quantum task planning operational")
            print("✅ Resource management functional")
            print("✅ System ready for Generation 2 enhancements")
            
            print("\n🚀 GENERATION 1 COMPLETE!")
            print("Next: Enhanced robustness, error handling, and security")
            
        else:
            print(f"\n⚠️ {total_tests - success_count} tests failed")
            print("🔧 Some components need attention before proceeding")
        
    except Exception as e:
        print(f"\n❌ Demo failed with exception: {e}")
        print("🔧 Check system configuration and dependencies")
        raise
    
    return success_count == total_tests


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)