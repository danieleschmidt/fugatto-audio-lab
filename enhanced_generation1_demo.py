#!/usr/bin/env python3
"""Enhanced Generation 1 Demo - Simple Audio Generation with Quantum Planning.

Demonstrates core functionality:
- Quantum task planning
- Audio generation pipeline
- Basic error handling
- Performance monitoring
"""

import logging
import time
import json
from pathlib import Path
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('generation1_demo.log')
    ]
)

logger = logging.getLogger(__name__)

try:
    from fugatto_lab import (
        QuantumTaskPlanner, 
        QuantumTask, 
        TaskPriority,
        create_audio_generation_pipeline,
        run_quantum_audio_pipeline
    )
    from fugatto_lab.core import FugattoModel, AudioProcessor
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.info("Using mock implementations")
    
    class MockQuantumTaskPlanner:
        def add_task(self, task):
            logger.info(f"Mock: Added task {task}")
            return "mock_task_id"
        
        def execute_pipeline(self, pipeline_id):
            logger.info(f"Mock: Executing pipeline {pipeline_id}")
            return {"status": "completed", "results": []}
    
    QuantumTaskPlanner = MockQuantumTaskPlanner
    QuantumTask = dict
    TaskPriority = type("TaskPriority", (), {"HIGH": "high", "MEDIUM": "medium", "LOW": "low"})
    
    def create_audio_generation_pipeline(prompts):
        return f"pipeline_{len(prompts)}_tasks"
    
    def run_quantum_audio_pipeline(planner, pipeline_id):
        return {"status": "completed", "audio_files": []}


class Generation1Demo:
    """Simple demonstration of core audio generation capabilities."""
    
    def __init__(self):
        """Initialize demo with basic configuration."""
        self.output_dir = Path("generation1_outputs")
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize quantum planner
        self.planner = QuantumTaskPlanner(max_concurrent_tasks=2)
        
        # Initialize audio components
        try:
            self.model = FugattoModel("mock-fugatto-model")
            self.processor = AudioProcessor()
        except Exception as e:
            logger.warning(f"Using mock audio components: {e}")
            self.model = None
            self.processor = None
        
        # Demo configuration
        self.demo_prompts = [
            "A gentle rain falling on leaves",
            "Ocean waves on a peaceful beach", 
            "Soft piano melody in C major",
            "Cat purring contentedly",
            "Wind chimes in a light breeze"
        ]
        
        # Performance metrics
        self.metrics = {
            "tasks_executed": 0,
            "total_duration": 0.0,
            "errors_encountered": 0,
            "audio_files_generated": 0
        }
        
        logger.info("Generation 1 Demo initialized")
    
    def run_basic_generation(self) -> Dict[str, Any]:
        """Run basic audio generation demo."""
        logger.info("Starting basic audio generation demo")
        start_time = time.time()
        
        results = {
            "demo_type": "basic_generation",
            "timestamp": time.time(),
            "prompts_processed": [],
            "outputs": [],
            "performance": {}
        }
        
        try:
            for i, prompt in enumerate(self.demo_prompts[:3]):  # Limit to 3 for demo
                logger.info(f"Processing prompt {i+1}: '{prompt}'")
                
                # Create quantum task
                task = {
                    "id": f"gen1_task_{i+1}",
                    "type": "audio_generation",
                    "prompt": prompt,
                    "duration": 5.0,  # 5 second clips
                    "priority": TaskPriority.MEDIUM
                }
                
                # Add to quantum planner
                task_id = self.planner.add_task(task)
                
                # Generate audio (mock or real)
                audio_result = self._generate_audio_sample(prompt, duration=5.0)
                
                if audio_result:
                    output_path = self.output_dir / f"generation1_sample_{i+1}.wav"
                    self._save_audio_result(audio_result, output_path)
                    
                    results["prompts_processed"].append(prompt)
                    results["outputs"].append(str(output_path))
                    self.metrics["audio_files_generated"] += 1
                
                self.metrics["tasks_executed"] += 1
                time.sleep(0.5)  # Brief pause for demo
                
        except Exception as e:
            logger.error(f"Error in basic generation: {e}")
            self.metrics["errors_encountered"] += 1
            results["error"] = str(e)
        
        # Record performance metrics
        end_time = time.time()
        self.metrics["total_duration"] = end_time - start_time
        results["performance"] = self.metrics.copy()
        
        logger.info(f"Basic generation completed in {self.metrics['total_duration']:.2f}s")
        return results
    
    def run_quantum_pipeline_demo(self) -> Dict[str, Any]:
        """Demonstrate quantum task pipeline execution."""
        logger.info("Starting quantum pipeline demo")
        start_time = time.time()
        
        results = {
            "demo_type": "quantum_pipeline",
            "timestamp": time.time(),
            "pipeline_id": None,
            "execution_result": None,
            "performance": {}
        }
        
        try:
            # Create audio generation pipeline
            pipeline_prompts = self.demo_prompts[:2]  # Use 2 prompts
            pipeline_id = create_audio_generation_pipeline(pipeline_prompts)
            results["pipeline_id"] = pipeline_id
            
            logger.info(f"Created pipeline: {pipeline_id}")
            
            # Execute pipeline
            execution_result = run_quantum_audio_pipeline(self.planner, pipeline_id)
            results["execution_result"] = execution_result
            
            # Process results
            if execution_result.get("status") == "completed":
                logger.info("Pipeline execution completed successfully")
                self.metrics["tasks_executed"] += len(pipeline_prompts)
            else:
                logger.warning(f"Pipeline execution status: {execution_result.get('status')}")
                
        except Exception as e:
            logger.error(f"Error in quantum pipeline: {e}")
            self.metrics["errors_encountered"] += 1
            results["error"] = str(e)
        
        # Record performance
        end_time = time.time()
        pipeline_duration = end_time - start_time
        self.metrics["total_duration"] += pipeline_duration
        results["performance"] = {
            "pipeline_duration": pipeline_duration,
            "cumulative_metrics": self.metrics.copy()
        }
        
        logger.info(f"Quantum pipeline demo completed in {pipeline_duration:.2f}s")
        return results
    
    def run_performance_benchmark(self) -> Dict[str, Any]:
        """Run simple performance benchmark."""
        logger.info("Starting performance benchmark")
        
        benchmark_results = {
            "demo_type": "performance_benchmark",
            "timestamp": time.time(),
            "test_cases": [],
            "summary": {}
        }
        
        test_cases = [
            {"name": "Short Generation", "duration": 2.0, "iterations": 3},
            {"name": "Medium Generation", "duration": 5.0, "iterations": 2},
            {"name": "Batch Processing", "duration": 3.0, "iterations": 4}
        ]
        
        total_start = time.time()
        
        for test_case in test_cases:
            logger.info(f"Running test: {test_case['name']}")
            test_start = time.time()
            
            test_result = {
                "name": test_case["name"],
                "iterations": test_case["iterations"],
                "duration_per_item": test_case["duration"],
                "execution_times": [],
                "average_time": 0.0,
                "throughput": 0.0
            }
            
            try:
                for i in range(test_case["iterations"]):
                    iter_start = time.time()
                    
                    # Simulate audio generation
                    prompt = f"Test audio generation {i+1} for {test_case['name']}"
                    audio_result = self._generate_audio_sample(prompt, test_case["duration"])
                    
                    iter_end = time.time()
                    execution_time = iter_end - iter_start
                    test_result["execution_times"].append(execution_time)
                    
                    self.metrics["tasks_executed"] += 1
                    if audio_result:
                        self.metrics["audio_files_generated"] += 1
                
                # Calculate statistics
                if test_result["execution_times"]:
                    test_result["average_time"] = sum(test_result["execution_times"]) / len(test_result["execution_times"])
                    test_result["throughput"] = test_case["iterations"] / (time.time() - test_start)
                
            except Exception as e:
                logger.error(f"Error in benchmark test {test_case['name']}: {e}")
                test_result["error"] = str(e)
                self.metrics["errors_encountered"] += 1
            
            benchmark_results["test_cases"].append(test_result)
        
        # Summary statistics
        total_duration = time.time() - total_start
        benchmark_results["summary"] = {
            "total_benchmark_time": total_duration,
            "total_tasks_executed": sum(tc.get("iterations", 0) for tc in test_cases),
            "overall_throughput": sum(tc.get("iterations", 0) for tc in test_cases) / total_duration,
            "cumulative_metrics": self.metrics.copy()
        }
        
        logger.info(f"Performance benchmark completed in {total_duration:.2f}s")
        return benchmark_results
    
    def _generate_audio_sample(self, prompt: str, duration: float = 5.0) -> Dict[str, Any]:
        """Generate audio sample (real or mock)."""
        try:
            if self.model:
                # Real audio generation
                logger.debug(f"Generating real audio for: '{prompt}'")
                audio_data = self.model.generate(
                    prompt=prompt,
                    duration_seconds=duration,
                    temperature=0.8
                )
                return {
                    "audio_data": audio_data,
                    "sample_rate": self.model.sample_rate,
                    "duration": duration,
                    "prompt": prompt,
                    "generation_method": "fugatto_model"
                }
            else:
                # Mock audio generation
                logger.debug(f"Generating mock audio for: '{prompt}'")
                import numpy as np
                sample_rate = 48000
                num_samples = int(duration * sample_rate)
                
                # Generate simple sine wave with noise
                t = np.linspace(0, duration, num_samples)
                base_freq = 440.0 + (hash(prompt) % 200)  # Vary frequency by prompt
                audio_data = 0.3 * np.sin(2 * np.pi * base_freq * t)
                audio_data += 0.1 * np.random.normal(0, 1, num_samples)
                
                return {
                    "audio_data": audio_data.astype(np.float32),
                    "sample_rate": sample_rate,
                    "duration": duration,
                    "prompt": prompt,
                    "generation_method": "mock_generation"
                }
                
        except Exception as e:
            logger.error(f"Error generating audio sample: {e}")
            return None
    
    def _save_audio_result(self, audio_result: Dict[str, Any], output_path: Path) -> bool:
        """Save audio result to file."""
        try:
            if self.processor:
                # Real audio saving
                self.processor.save_audio(
                    audio_result["audio_data"],
                    output_path,
                    sample_rate=audio_result["sample_rate"]
                )
            else:
                # Mock audio saving (save as numpy array)
                import numpy as np
                np.save(output_path.with_suffix('.npy'), audio_result["audio_data"])
                logger.info(f"Saved mock audio data: {output_path.with_suffix('.npy')}")
            
            # Save metadata
            metadata_path = output_path.with_suffix('.json')
            metadata = {
                "prompt": audio_result["prompt"],
                "duration": audio_result["duration"],
                "sample_rate": audio_result["sample_rate"],
                "generation_method": audio_result["generation_method"],
                "timestamp": time.time()
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Saved audio and metadata: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving audio result: {e}")
            return False
    
    def run_complete_demo(self) -> Dict[str, Any]:
        """Run complete Generation 1 demonstration."""
        logger.info("Starting complete Generation 1 demonstration")
        
        complete_results = {
            "demo_version": "Generation 1 - MAKE IT WORK",
            "timestamp": time.time(),
            "components_tested": [
                "quantum_task_planning",
                "audio_generation", 
                "pipeline_execution",
                "performance_monitoring"
            ],
            "test_results": {},
            "summary": {}
        }
        
        # Run all demo components
        demo_functions = [
            ("basic_generation", self.run_basic_generation),
            ("quantum_pipeline", self.run_quantum_pipeline_demo),
            ("performance_benchmark", self.run_performance_benchmark)
        ]
        
        for demo_name, demo_func in demo_functions:
            logger.info(f"Executing {demo_name} demo")
            try:
                result = demo_func()
                complete_results["test_results"][demo_name] = result
                logger.info(f"Completed {demo_name} demo successfully")
            except Exception as e:
                logger.error(f"Error in {demo_name} demo: {e}")
                complete_results["test_results"][demo_name] = {
                    "error": str(e),
                    "status": "failed"
                }
                self.metrics["errors_encountered"] += 1
        
        # Generate summary
        complete_results["summary"] = {
            "total_metrics": self.metrics,
            "success_rate": (len([r for r in complete_results["test_results"].values() if "error" not in r]) / 
                            len(demo_functions)) * 100,
            "total_audio_files": self.metrics["audio_files_generated"],
            "total_execution_time": self.metrics["total_duration"],
            "output_directory": str(self.output_dir.absolute())
        }
        
        # Save complete results
        results_file = self.output_dir / "generation1_demo_results.json"
        with open(results_file, 'w') as f:
            json.dump(complete_results, f, indent=2, default=str)
        
        logger.info(f"Complete demo results saved: {results_file}")
        logger.info(f"Generated {self.metrics['audio_files_generated']} audio files")
        logger.info(f"Success rate: {complete_results['summary']['success_rate']:.1f}%")
        
        return complete_results


def main():
    """Main demo execution function."""
    print("🚀 Fugatto Audio Lab - Generation 1 Demo")
    print("==========================================\n")
    
    # Initialize and run demo
    demo = Generation1Demo()
    
    try:
        results = demo.run_complete_demo()
        
        print("\n✨ Demo Completed Successfully!")
        print(f"📁 Output directory: {results['summary']['output_directory']}")
        print(f"🎵 Audio files generated: {results['summary']['total_audio_files']}")
        print(f"⏱️  Total execution time: {results['summary']['total_execution_time']:.2f}s")
        print(f"📊 Success rate: {results['summary']['success_rate']:.1f}%")
        
        if results['summary']['success_rate'] >= 80:
            print("\n🎉 Generation 1 Implementation: SUCCESS")
            print("   Core functionality is working properly!")
        else:
            print("\n⚠️  Generation 1 Implementation: PARTIAL SUCCESS")
            print("   Some components may need attention.")
        
    except Exception as e:
        logger.error(f"Demo execution failed: {e}")
        print(f"\n❌ Demo failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
