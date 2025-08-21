#!/usr/bin/env python3
"""Simple Generation Demo for Fugatto Audio Lab.

Demonstrates the core audio generation capabilities with minimal setup.
This demo showcases Generation 1 functionality - making it work with simple, effective features.
"""

import sys
import logging
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from fugatto_lab.simple_api import SimpleAudioAPI, generate, demo
    from fugatto_lab.batch_processor import BatchProcessor
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demo_simple_generation():
    """Demonstrate simple audio generation."""
    print("🎵 Fugatto Audio Lab - Simple Generation Demo")
    print("=" * 50)
    
    # Initialize API
    print("\n1. Initializing Simple Audio API...")
    api = SimpleAudioAPI()
    
    # Show model status
    status = api.get_model_status()
    print(f"   Model: {status['model_name']}")
    print(f"   Device: {status['device']}")
    print(f"   Audio Core Available: {status['features_available']['audio_core']}")
    print(f"   Output Directory: {status['output_directory']}")
    
    # Test basic generation
    print("\n2. Testing Basic Audio Generation...")
    
    test_prompts = [
        ("A cat meowing softly", 3.0),
        ("Rain falling on a roof", 4.0),
        ("Bird singing at dawn", 2.5),
        ("Ocean waves crashing", 5.0)
    ]
    
    results = []
    
    for prompt, duration in test_prompts:
        print(f"\n   Generating: '{prompt}' ({duration}s)")
        
        try:
            result = api.generate_audio(
                prompt=prompt,
                duration=duration,
                temperature=0.8
            )
            
            if result['status'] == 'completed':
                print(f"   ✅ Success: {result['output_path']}")
                print(f"      Generation time: {result['generation_time']:.2f}s")
                print(f"      Real-time factor: {result['real_time_factor']:.2f}x")
                
                # Show audio stats if available
                if 'audio_stats' in result:
                    stats = result['audio_stats']
                    print(f"      Audio duration: {stats.get('duration_seconds', 'N/A'):.2f}s")
                    print(f"      Peak level: {stats.get('peak', 'N/A'):.3f}")
            else:
                print(f"   ❌ Failed: {result.get('error', 'Unknown error')}")
            
            results.append(result)
            
        except Exception as e:
            print(f"   ❌ Exception: {e}")
            results.append({'status': 'failed', 'error': str(e)})
    
    # Show summary
    successful = sum(1 for r in results if r.get('status') == 'completed')
    print(f"\n3. Generation Summary:")
    print(f"   Total attempts: {len(results)}")
    print(f"   Successful: {successful}")
    print(f"   Failed: {len(results) - successful}")
    
    if successful > 0:
        avg_time = sum(r.get('generation_time', 0) for r in results if r.get('status') == 'completed') / successful
        print(f"   Average generation time: {avg_time:.2f}s")
    
    return results


def demo_batch_processing():
    """Demonstrate batch processing capabilities."""
    print("\n\n🔄 Batch Processing Demo")
    print("=" * 50)
    
    # Initialize batch processor
    print("\n1. Setting up Batch Processor...")
    processor = BatchProcessor(max_workers=2)
    
    # Add multiple tasks
    print("\n2. Adding Batch Tasks...")
    batch_prompts = [
        "A guitar playing a gentle melody",
        "Wind blowing through trees",
        "Children laughing and playing",
        "Coffee shop ambiance with chatter",
        "Fireplace crackling with warmth"
    ]
    
    task_ids = []
    for i, prompt in enumerate(batch_prompts):
        task_id = f"batch_task_{i+1}"
        processor.add_generation_task(
            task_id=task_id,
            prompt=prompt,
            duration=3.0,
            priority=i % 3  # Vary priorities
        )
        task_ids.append(task_id)
        print(f"   Added: {task_id} - '{prompt}'")
    
    # Start processing
    print(f"\n3. Processing {len(task_ids)} tasks...")
    processor.start_processing()
    
    # Monitor progress
    import time
    while processor.is_processing:
        progress = processor.get_progress()
        print(f"   Progress: {progress.completion_rate:.1f}% "
              f"({progress.completed_tasks}/{progress.total_tasks}) "
              f"[{progress.processing_tasks} active]")
        time.sleep(2)
    
    # Show results
    print("\n4. Batch Results:")
    final_progress = processor.get_progress()
    
    for task_id in task_ids:
        task_status = processor.get_task_status(task_id)
        if task_status:
            status_icon = "✅" if task_status['status'] == 'completed' else "❌"
            print(f"   {status_icon} {task_id}: {task_status['status']}")
            if task_status['processing_time']:
                print(f"      Processing time: {task_status['processing_time']:.2f}s")
    
    print(f"\n   Batch Summary:")
    print(f"   ✅ Completed: {final_progress.completed_tasks}")
    print(f"   ❌ Failed: {final_progress.failed_tasks}")
    print(f"   ⏱️  Total time: {final_progress.elapsed_time:.2f}s")
    print(f"   📊 Average per task: {final_progress.average_task_time:.2f}s")
    
    return processor


def demo_quick_functions():
    """Demonstrate quick convenience functions."""
    print("\n\n⚡ Quick Functions Demo")
    print("=" * 50)
    
    # Test quick generate function
    print("\n1. Testing Quick Generate Function...")
    try:
        output_path = generate(
            prompt="A short electronic beep",
            duration=1.0
        )
        print(f"   ✅ Quick generation successful: {output_path}")
    except Exception as e:
        print(f"   ❌ Quick generation failed: {e}")
    
    # Run built-in demo
    print("\n2. Running Built-in Demo...")
    try:
        demo_results = demo()
        print(f"   ✅ Built-in demo completed")
        print(f"   API Status: {demo_results['api_status']['features_available']}")
        print(f"   Successful generations: {demo_results['successful_generations']}")
        
        for gen in demo_results['demo_generations']:
            status_icon = "✅" if gen['success'] else "❌"
            print(f"   {status_icon} '{gen['prompt']}' - {gen.get('duration', 0):.2f}s")
    
    except Exception as e:
        print(f"   ❌ Built-in demo failed: {e}")


def demo_file_management():
    """Demonstrate file management capabilities."""
    print("\n\n📁 File Management Demo")
    print("=" * 50)
    
    api = SimpleAudioAPI()
    
    # List outputs
    print("\n1. Listing Recent Outputs...")
    outputs = api.list_outputs(limit=10)
    
    if outputs:
        print(f"   Found {len(outputs)} output files:")
        for output in outputs[:5]:  # Show first 5
            size_kb = output['size_bytes'] / 1024
            print(f"   📄 {output['filename']} ({size_kb:.1f} KB)")
    else:
        print("   No output files found")
    
    # Show configuration
    print("\n2. Current Configuration:")
    config = api.config
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    # Demonstrate configuration update
    print("\n3. Updating Configuration...")
    new_config = api.configure(
        default_temperature=0.9,
        max_duration=20.0
    )
    print(f"   Updated temperature: {new_config['default_temperature']}")
    print(f"   Updated max duration: {new_config['max_duration']}")


def main():
    """Run complete demo suite."""
    print("🚀 Fugatto Audio Lab - Generation 1 Demo Suite")
    print("=" * 60)
    print("Demonstrating core functionality: Simple, Fast, Effective")
    print()
    
    try:
        # Run all demos
        demo_simple_generation()
        demo_batch_processing()
        demo_quick_functions()
        demo_file_management()
        
        print("\n\n🎉 Demo Suite Completed Successfully!")
        print("=" * 60)
        print("\nGeneration 1 Features Demonstrated:")
        print("✅ Simple audio generation API")
        print("✅ Batch processing engine")
        print("✅ Progress tracking and monitoring")
        print("✅ Quick convenience functions")
        print("✅ File management utilities")
        print("✅ Graceful error handling")
        print("✅ Configurable parameters")
        print("\nNext: Generation 2 will add robust validation and enterprise features!")
        
    except Exception as e:
        logger.error(f"Demo suite failed: {e}")
        print(f"\n❌ Demo suite failed with error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)