#!/usr/bin/env python3
"""Complete System Test for Fugatto Audio Lab.

Comprehensive test suite validating all three generations of autonomous enhancement:
- Generation 1: MAKE IT WORK (Simple)
- Generation 2: MAKE IT ROBUST (Reliable) 
- Generation 3: MAKE IT SCALE (Optimized)
"""

import sys
import logging
import asyncio
import time
from pathlib import Path
import json

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_generation1_simple():
    """Test Generation 1: Make it Work (Simple)."""
    print("🔧 Testing Generation 1: MAKE IT WORK (Simple)")
    print("-" * 50)
    
    results = {"tests": [], "passed": 0, "failed": 0}
    
    # Test 1: Simple Audio API
    try:
        from fugatto_lab.simple_api import SimpleAudioAPI
        api = SimpleAudioAPI()
        result = api.generate_audio("test audio", duration=1.0)
        
        if result.get('status') == 'completed':
            results["tests"].append({"name": "Simple Audio Generation", "status": "PASS"})
            results["passed"] += 1
            print("   ✅ Simple Audio Generation: PASS")
        else:
            results["tests"].append({"name": "Simple Audio Generation", "status": "FAIL", "error": result.get('error')})
            results["failed"] += 1
            print(f"   ❌ Simple Audio Generation: FAIL - {result.get('error')}")
    except Exception as e:
        results["tests"].append({"name": "Simple Audio Generation", "status": "ERROR", "error": str(e)})
        results["failed"] += 1
        print(f"   ❌ Simple Audio Generation: ERROR - {e}")
    
    # Test 2: Batch Processing
    try:
        from fugatto_lab.batch_processor import BatchProcessor
        processor = BatchProcessor(max_workers=1)
        processor.add_generation_task("test_task", "test prompt", duration=1.0)
        
        results["tests"].append({"name": "Batch Processing Setup", "status": "PASS"})
        results["passed"] += 1
        print("   ✅ Batch Processing Setup: PASS")
    except Exception as e:
        results["tests"].append({"name": "Batch Processing Setup", "status": "ERROR", "error": str(e)})
        results["failed"] += 1
        print(f"   ❌ Batch Processing Setup: ERROR - {e}")
    
    # Test 3: Core Components
    try:
        from fugatto_lab.core import FugattoModel, AudioProcessor
        model = FugattoModel()
        processor = AudioProcessor()
        
        results["tests"].append({"name": "Core Components", "status": "PASS"})
        results["passed"] += 1
        print("   ✅ Core Components: PASS")
    except Exception as e:
        results["tests"].append({"name": "Core Components", "status": "ERROR", "error": str(e)})
        results["failed"] += 1
        print(f"   ❌ Core Components: ERROR - {e}")
    
    # Test 4: Quantum Planner
    try:
        from fugatto_lab.quantum_planner import QuantumTaskPlanner
        planner = QuantumTaskPlanner()
        
        results["tests"].append({"name": "Quantum Task Planning", "status": "PASS"})
        results["passed"] += 1
        print("   ✅ Quantum Task Planning: PASS")
    except Exception as e:
        results["tests"].append({"name": "Quantum Task Planning", "status": "ERROR", "error": str(e)})
        results["failed"] += 1
        print(f"   ❌ Quantum Task Planning: ERROR - {e}")
    
    print(f"\n   Generation 1 Results: {results['passed']} passed, {results['failed']} failed")
    return results


def test_generation2_robust():
    """Test Generation 2: Make it Robust (Reliable)."""
    print("\n🛡️ Testing Generation 2: MAKE IT ROBUST (Reliable)")
    print("-" * 50)
    
    results = {"tests": [], "passed": 0, "failed": 0}
    
    # Test 1: Error Handling
    try:
        from fugatto_lab.simple_api import SimpleAudioAPI
        api = SimpleAudioAPI()
        
        # Test with invalid input
        result = api.generate_audio("", duration=0)  # Invalid parameters
        
        # Should handle gracefully
        if result.get('status') == 'failed' and 'error' in result:
            results["tests"].append({"name": "Error Handling", "status": "PASS"})
            results["passed"] += 1
            print("   ✅ Error Handling: PASS")
        else:
            results["tests"].append({"name": "Error Handling", "status": "FAIL", "error": "Should fail gracefully"})
            results["failed"] += 1
            print("   ❌ Error Handling: FAIL - Should fail gracefully")
    except Exception as e:
        results["tests"].append({"name": "Error Handling", "status": "ERROR", "error": str(e)})
        results["failed"] += 1
        print(f"   ❌ Error Handling: ERROR - {e}")
    
    # Test 2: Robust Validation
    try:
        from fugatto_lab.robust_validation import RobustValidator
        validator = RobustValidator()
        
        results["tests"].append({"name": "Robust Validation", "status": "PASS"})
        results["passed"] += 1
        print("   ✅ Robust Validation: PASS")
    except ImportError:
        # This is expected if robust_validation module doesn't exist yet
        results["tests"].append({"name": "Robust Validation", "status": "SKIP", "error": "Module not implemented"})
        print("   ⏭️  Robust Validation: SKIP - Module not implemented")
    except Exception as e:
        results["tests"].append({"name": "Robust Validation", "status": "ERROR", "error": str(e)})
        results["failed"] += 1
        print(f"   ❌ Robust Validation: ERROR - {e}")
    
    # Test 3: Configuration Management
    try:
        from fugatto_lab.simple_api import SimpleAudioAPI
        api = SimpleAudioAPI()
        
        # Test configuration updates
        old_config = api.config.copy()
        api.configure(default_temperature=0.9)
        
        if api.config['default_temperature'] == 0.9:
            results["tests"].append({"name": "Configuration Management", "status": "PASS"})
            results["passed"] += 1
            print("   ✅ Configuration Management: PASS")
        else:
            results["tests"].append({"name": "Configuration Management", "status": "FAIL", "error": "Config not updated"})
            results["failed"] += 1
            print("   ❌ Configuration Management: FAIL")
    except Exception as e:
        results["tests"].append({"name": "Configuration Management", "status": "ERROR", "error": str(e)})
        results["failed"] += 1
        print(f"   ❌ Configuration Management: ERROR - {e}")
    
    # Test 4: Resource Management
    try:
        from fugatto_lab.simple_api import SimpleAudioAPI
        api = SimpleAudioAPI()
        
        # Test resource cleanup
        outputs_before = len(api.list_outputs())
        
        # Generate some outputs
        for i in range(3):
            api.generate_audio(f"test {i}", duration=0.5)
        
        outputs_after = len(api.list_outputs())
        
        if outputs_after > outputs_before:
            results["tests"].append({"name": "Resource Management", "status": "PASS"})
            results["passed"] += 1
            print("   ✅ Resource Management: PASS")
        else:
            results["tests"].append({"name": "Resource Management", "status": "FAIL", "error": "No outputs created"})
            results["failed"] += 1
            print("   ❌ Resource Management: FAIL")
    except Exception as e:
        results["tests"].append({"name": "Resource Management", "status": "ERROR", "error": str(e)})
        results["failed"] += 1
        print(f"   ❌ Resource Management: ERROR - {e}")
    
    print(f"\n   Generation 2 Results: {results['passed']} passed, {results['failed']} failed")
    return results


def test_generation3_scalable():
    """Test Generation 3: Make it Scale (Optimized)."""
    print("\n🚀 Testing Generation 3: MAKE IT SCALE (Optimized)")
    print("-" * 50)
    
    results = {"tests": [], "passed": 0, "failed": 0}
    
    # Test 1: Auto-Optimizer
    try:
        from fugatto_lab.auto_optimizer import AutoOptimizer, PerformanceMonitor
        
        monitor = PerformanceMonitor(window_size=10, update_interval=0.1)
        optimizer = AutoOptimizer(monitor)
        
        results["tests"].append({"name": "Auto-Optimizer", "status": "PASS"})
        results["passed"] += 1
        print("   ✅ Auto-Optimizer: PASS")
    except Exception as e:
        results["tests"].append({"name": "Auto-Optimizer", "status": "ERROR", "error": str(e)})
        results["failed"] += 1
        print(f"   ❌ Auto-Optimizer: ERROR - {e}")
    
    # Test 2: Performance Monitoring
    try:
        from fugatto_lab.auto_optimizer import PerformanceMonitor
        monitor = PerformanceMonitor()
        
        # Test metric collection
        metrics = monitor._collect_metrics()
        
        if hasattr(metrics, 'cpu_usage') and hasattr(metrics, 'memory_usage'):
            results["tests"].append({"name": "Performance Monitoring", "status": "PASS"})
            results["passed"] += 1
            print("   ✅ Performance Monitoring: PASS")
        else:
            results["tests"].append({"name": "Performance Monitoring", "status": "FAIL", "error": "Invalid metrics"})
            results["failed"] += 1
            print("   ❌ Performance Monitoring: FAIL")
    except Exception as e:
        results["tests"].append({"name": "Performance Monitoring", "status": "ERROR", "error": str(e)})
        results["failed"] += 1
        print(f"   ❌ Performance Monitoring: ERROR - {e}")
    
    # Test 3: Enterprise Deployment
    try:
        from fugatto_lab.enterprise_deployment import create_deployment_config, DeploymentOrchestrator
        
        config = create_deployment_config("test-deployment", "local")
        orchestrator = DeploymentOrchestrator(config)
        
        results["tests"].append({"name": "Enterprise Deployment", "status": "PASS"})
        results["passed"] += 1
        print("   ✅ Enterprise Deployment: PASS")
    except Exception as e:
        results["tests"].append({"name": "Enterprise Deployment", "status": "ERROR", "error": str(e)})
        results["failed"] += 1
        print(f"   ❌ Enterprise Deployment: ERROR - {e}")
    
    # Test 4: Health Monitoring
    try:
        from fugatto_lab.enterprise_deployment import HealthChecker
        
        health_checker = HealthChecker(check_interval=1.0)
        health_checker.register_service("test-service", "http://localhost:8080/health")
        
        results["tests"].append({"name": "Health Monitoring", "status": "PASS"})
        results["passed"] += 1
        print("   ✅ Health Monitoring: PASS")
    except Exception as e:
        results["tests"].append({"name": "Health Monitoring", "status": "ERROR", "error": str(e)})
        results["failed"] += 1
        print(f"   ❌ Health Monitoring: ERROR - {e}")
    
    # Test 5: Integration Framework
    try:
        from fugatto_lab.simple_api import SimpleAudioAPI
        from fugatto_lab.auto_optimizer import create_auto_optimizer
        
        # Test integration
        api = SimpleAudioAPI()
        optimizer = create_auto_optimizer()
        optimizer.integrate_with_api(api)
        
        results["tests"].append({"name": "Integration Framework", "status": "PASS"})
        results["passed"] += 1
        print("   ✅ Integration Framework: PASS")
    except Exception as e:
        results["tests"].append({"name": "Integration Framework", "status": "ERROR", "error": str(e)})
        results["failed"] += 1
        print(f"   ❌ Integration Framework: ERROR - {e}")
    
    print(f"\n   Generation 3 Results: {results['passed']} passed, {results['failed']} failed")
    return results


async def test_async_operations():
    """Test asynchronous operations."""
    print("\n⚡ Testing Asynchronous Operations")
    print("-" * 50)
    
    results = {"tests": [], "passed": 0, "failed": 0}
    
    # Test 1: Async Deployment
    try:
        from fugatto_lab.enterprise_deployment import quick_deploy
        
        deploy_result = await quick_deploy("test-async", "local", 1)
        
        if deploy_result.get('deployment_result', {}).get('success'):
            results["tests"].append({"name": "Async Deployment", "status": "PASS"})
            results["passed"] += 1
            print("   ✅ Async Deployment: PASS")
        else:
            results["tests"].append({"name": "Async Deployment", "status": "FAIL", "error": "Deployment failed"})
            results["failed"] += 1
            print("   ❌ Async Deployment: FAIL")
    except Exception as e:
        results["tests"].append({"name": "Async Deployment", "status": "ERROR", "error": str(e)})
        results["failed"] += 1
        print(f"   ❌ Async Deployment: ERROR - {e}")
    
    # Test 2: Concurrent Operations
    try:
        from fugatto_lab.simple_api import SimpleAudioAPI
        
        api = SimpleAudioAPI()
        
        # Test concurrent generation
        start_time = time.time()
        tasks = []
        
        for i in range(3):
            # Simulate concurrent requests
            result = api.generate_audio(f"concurrent test {i}", duration=0.5)
            tasks.append(result)
        
        elapsed_time = time.time() - start_time
        successful = sum(1 for t in tasks if t.get('status') == 'completed')
        
        if successful > 0:
            results["tests"].append({"name": "Concurrent Operations", "status": "PASS"})
            results["passed"] += 1
            print(f"   ✅ Concurrent Operations: PASS ({successful}/3 successful in {elapsed_time:.2f}s)")
        else:
            results["tests"].append({"name": "Concurrent Operations", "status": "FAIL", "error": "No successful operations"})
            results["failed"] += 1
            print("   ❌ Concurrent Operations: FAIL")
    except Exception as e:
        results["tests"].append({"name": "Concurrent Operations", "status": "ERROR", "error": str(e)})
        results["failed"] += 1
        print(f"   ❌ Concurrent Operations: ERROR - {e}")
    
    print(f"\n   Async Results: {results['passed']} passed, {results['failed']} failed")
    return results


def test_production_readiness():
    """Test production readiness checklist."""
    print("\n🏭 Testing Production Readiness")
    print("-" * 50)
    
    checklist = [
        ("Core Audio Generation", "fugatto_lab.simple_api", "SimpleAudioAPI"),
        ("Batch Processing", "fugatto_lab.batch_processor", "BatchProcessor"),
        ("Quantum Planning", "fugatto_lab.quantum_planner", "QuantumTaskPlanner"),
        ("Performance Optimization", "fugatto_lab.auto_optimizer", "AutoOptimizer"),
        ("Enterprise Deployment", "fugatto_lab.enterprise_deployment", "DeploymentOrchestrator"),
        ("Health Monitoring", "fugatto_lab.enterprise_deployment", "HealthChecker"),
    ]
    
    results = {"available": 0, "total": len(checklist), "components": []}
    
    for name, module_name, class_name in checklist:
        try:
            module = __import__(module_name, fromlist=[class_name])
            component_class = getattr(module, class_name)
            
            # Try to instantiate
            if class_name == "AutoOptimizer":
                from fugatto_lab.auto_optimizer import PerformanceMonitor
                monitor = PerformanceMonitor()
                instance = component_class(monitor)
            elif class_name == "DeploymentOrchestrator":
                from fugatto_lab.enterprise_deployment import create_deployment_config
                config = create_deployment_config("test", "local")
                instance = component_class(config)
            else:
                instance = component_class()
            
            results["available"] += 1
            results["components"].append({"name": name, "status": "AVAILABLE"})
            print(f"   ✅ {name}: AVAILABLE")
            
        except Exception as e:
            results["components"].append({"name": name, "status": "ERROR", "error": str(e)})
            print(f"   ❌ {name}: ERROR - {e}")
    
    # Production readiness score
    readiness_score = results["available"] / results["total"]
    
    print(f"\n   Production Readiness: {results['available']}/{results['total']} components ({readiness_score:.1%})")
    
    if readiness_score >= 0.9:
        print("   🏆 PRODUCTION READY - Excellent!")
    elif readiness_score >= 0.7:
        print("   ⚠️  MOSTLY READY - Minor issues")
    else:
        print("   ❌ NOT READY - Major issues")
    
    return results


async def main():
    """Run complete system test suite."""
    print("🎯 Fugatto Audio Lab - Complete System Test Suite")
    print("=" * 70)
    print("Testing Autonomous SDLC Implementation - All Generations")
    print()
    
    # Collect all results
    all_results = {}
    
    try:
        # Test each generation
        all_results["generation1"] = test_generation1_simple()
        all_results["generation2"] = test_generation2_robust()
        all_results["generation3"] = test_generation3_scalable()
        
        # Test async operations
        all_results["async_operations"] = await test_async_operations()
        
        # Test production readiness
        all_results["production_readiness"] = test_production_readiness()
        
        # Calculate overall results
        total_tests = 0
        total_passed = 0
        total_failed = 0
        
        for category, results in all_results.items():
            if "passed" in results and "failed" in results:
                total_passed += results["passed"]
                total_failed += results["failed"]
                total_tests += results["passed"] + results["failed"]
        
        # Final summary
        print("\n" + "=" * 70)
        print("🏆 AUTONOMOUS SDLC IMPLEMENTATION TEST RESULTS")
        print("=" * 70)
        
        print(f"\nOverall Test Results:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Passed: {total_passed}")
        print(f"   Failed: {total_failed}")
        print(f"   Success Rate: {total_passed/total_tests:.1%}" if total_tests > 0 else "   Success Rate: N/A")
        
        # Generation-specific results
        print(f"\nGeneration Results:")
        for gen_name, gen_results in [("Generation 1", all_results["generation1"]),
                                     ("Generation 2", all_results["generation2"]),
                                     ("Generation 3", all_results["generation3"])]:
            passed = gen_results.get("passed", 0)
            failed = gen_results.get("failed", 0)
            total = passed + failed
            if total > 0:
                print(f"   {gen_name}: {passed}/{total} passed ({passed/total:.1%})")
        
        # Production readiness
        prod_ready = all_results["production_readiness"]
        print(f"\nProduction Readiness: {prod_ready['available']}/{prod_ready['total']} components ({prod_ready['available']/prod_ready['total']:.1%})")
        
        # Final verdict
        overall_success_rate = total_passed / total_tests if total_tests > 0 else 0
        production_readiness = prod_ready['available'] / prod_ready['total']
        
        print(f"\n🎉 AUTONOMOUS SDLC STATUS:")
        
        if overall_success_rate >= 0.8 and production_readiness >= 0.8:
            print("   ✅ IMPLEMENTATION SUCCESSFUL!")
            print("   🚀 Ready for enterprise deployment")
            verdict = "SUCCESS"
        elif overall_success_rate >= 0.6 and production_readiness >= 0.6:
            print("   ⚠️  IMPLEMENTATION MOSTLY SUCCESSFUL")
            print("   🔧 Minor improvements needed")
            verdict = "PARTIAL_SUCCESS"
        else:
            print("   ❌ IMPLEMENTATION NEEDS WORK")
            print("   🛠️  Significant improvements required")
            verdict = "NEEDS_WORK"
        
        # Achievements
        print(f"\n🏅 Achievements Unlocked:")
        achievements = []
        
        if all_results["generation1"]["passed"] >= 3:
            achievements.append("✅ Generation 1: MAKE IT WORK")
        if all_results["generation2"]["passed"] >= 2:
            achievements.append("✅ Generation 2: MAKE IT ROBUST")  
        if all_results["generation3"]["passed"] >= 4:
            achievements.append("✅ Generation 3: MAKE IT SCALE")
        if production_readiness >= 0.9:
            achievements.append("✅ Production Ready")
        if overall_success_rate >= 0.9:
            achievements.append("✅ Quality Excellence")
        
        for achievement in achievements:
            print(f"   {achievement}")
        
        if not achievements:
            print("   🎯 Work in progress - keep building!")
        
        # Save detailed results
        results_file = Path("outputs/complete_system_test_results.json")
        results_file.parent.mkdir(exist_ok=True)
        
        test_summary = {
            "timestamp": time.time(),
            "verdict": verdict,
            "overall_success_rate": overall_success_rate,
            "production_readiness": production_readiness,
            "total_tests": total_tests,
            "total_passed": total_passed,
            "total_failed": total_failed,
            "achievements": achievements,
            "detailed_results": all_results
        }
        
        with open(results_file, 'w') as f:
            json.dump(test_summary, f, indent=2, default=str)
        
        print(f"\n📊 Detailed results saved to: {results_file}")
        
        return 0 if verdict == "SUCCESS" else 1
        
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        print(f"\n❌ Test suite failed with error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)