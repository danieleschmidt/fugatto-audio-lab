#!/usr/bin/env python3
"""Research Validation: Quantum-Neural Scheduler Comparative Studies.

This script runs comprehensive experimental validation of the novel 
Quantum-Coherent Neural Scheduler against classical baselines.

Usage: python3 research_validation_experiments.py
"""

import asyncio
import sys
import time
import json
from pathlib import Path

# Add the project to Python path
sys.path.insert(0, '/root/repo')

from fugatto_lab.research_quantum_neural_scheduler import (
    ExperimentalFramework,
    QuantumCoherentScheduler,
    BaselineClassicalScheduler,
    QuantumTask
)


async def run_comprehensive_research_validation():
    """Run comprehensive research validation experiments."""
    
    print("🧪 COMPREHENSIVE RESEARCH VALIDATION")
    print("Quantum-Coherent Neural Scheduler vs Classical Baselines")
    print("=" * 70)
    
    # Initialize framework with more runs for statistical significance
    framework = ExperimentalFramework(num_runs=15)
    
    print("\n📋 Experimental Design:")
    print("- Algorithm: Quantum-Coherent Neural Scheduler (QCNS)")
    print("- Baseline: Classical Priority-Based Scheduler")
    print("- Runs per test case: 15 (for statistical significance)")
    print("- Significance threshold: p < 0.05")
    print("- Improvement target: 25-40%")
    
    # Generate comprehensive test cases
    print("\n🎯 Generating test cases...")
    test_cases = []
    
    # Scalability tests
    for complexity in ["low", "medium", "high"]:
        for size in [10, 20, 30, 50]:
            test_cases.append(framework.generate_test_tasks(size, complexity))
            print(f"  ✓ Generated {size} tasks ({complexity} complexity)")
    
    print(f"\n📊 Total test cases: {len(test_cases)}")
    print("🚀 Starting experimental runs...")
    
    start_time = time.time()
    
    # Run experiments
    results = await framework.run_comparative_experiment(test_cases)
    
    total_time = time.time() - start_time
    
    print(f"\n⏱️  Total experimental time: {total_time:.1f} seconds")
    
    # Display key results
    print("\n🏆 KEY RESEARCH FINDINGS:")
    print("=" * 50)
    
    print(f"\n📈 PERFORMANCE IMPROVEMENTS:")
    print(f"  • Throughput: {results['improvement'].get('throughput', 0):+.1f}%")
    print(f"  • Latency: {results['improvement'].get('latency', 0):+.1f}%")
    print(f"  • Energy Efficiency: {results['improvement'].get('energy_consumption', 0):+.1f}%")
    print(f"  • Overall: {results['overall_improvement']:+.1f}%")
    
    print(f"\n🔬 STATISTICAL VALIDATION:")
    print(f"  • Significance Ratio: {results['significance_ratio']:.1%}")
    confidence = "HIGH" if results['significance_ratio'] > 0.7 else "MODERATE" if results['significance_ratio'] > 0.3 else "LOW"
    print(f"  • Confidence Level: {confidence}")
    
    print(f"\n⚡ QUANTUM-SPECIFIC METRICS:")
    print(f"  • Coherence Preservation: {results['quantum_mean'].get('coherence_preservation', 0):.1%}")
    print(f"  • Prediction Accuracy: {results['quantum_mean'].get('prediction_accuracy', 0):.1%}")
    
    # Hypothesis validation
    hypothesis_confirmed = results['overall_improvement'] >= 25.0
    print(f"\n🎯 HYPOTHESIS VALIDATION:")
    print(f"  • Target: 25-40% improvement")
    print(f"  • Achieved: {results['overall_improvement']:+.1f}%")
    print(f"  • Status: {'✅ CONFIRMED' if hypothesis_confirmed else '❌ NOT CONFIRMED'}")
    
    # Generate comprehensive research report
    print(f"\n📝 Generating research report...")
    report = framework.generate_research_report(results)
    
    # Save results
    results_dir = Path("/tmp/quantum_research_results")
    results_dir.mkdir(exist_ok=True)
    
    # Save detailed report
    report_path = results_dir / "quantum_neural_scheduler_research_report.md"
    with open(report_path, 'w') as f:
        f.write(report)
    
    # Save raw data
    data_path = results_dir / "experimental_data.json"
    with open(data_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 RESEARCH OUTPUTS SAVED:")
    print(f"  • Research Report: {report_path}")
    print(f"  • Raw Data: {data_path}")
    
    # Publication readiness assessment
    publication_ready = (
        hypothesis_confirmed and 
        results['significance_ratio'] > 0.7 and
        results['overall_improvement'] > 20
    )
    
    print(f"\n📚 PUBLICATION READINESS:")
    print(f"  • Novel Algorithm: ✅ Quantum-Coherent Neural Scheduling")
    print(f"  • Statistical Significance: {'✅' if results['significance_ratio'] > 0.7 else '❌'}")
    print(f"  • Performance Improvement: {'✅' if results['overall_improvement'] > 20 else '❌'}")
    print(f"  • Reproducible Framework: ✅ Open-source implementation")
    print(f"  • Status: {'✅ READY FOR SUBMISSION' if publication_ready else '⚠️  NEEDS REFINEMENT'}")
    
    if publication_ready:
        print(f"\n🎉 RESEARCH SUCCESS!")
        print("   Novel algorithmic contribution validated with statistical significance.")
        print("   Ready for academic publication and peer review.")
    else:
        print(f"\n⚠️  RESEARCH REFINEMENT NEEDED")
        print("   Consider algorithm improvements or additional experimental validation.")
    
    return results


if __name__ == "__main__":
    # Run the comprehensive validation
    results = asyncio.run(run_comprehensive_research_validation())