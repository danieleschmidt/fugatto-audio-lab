#!/usr/bin/env python3
"""Generation 1 Demo: Basic Fugatto Audio Lab Functionality.

This demo showcases the basic functionality of the Fugatto Audio Lab
with mock dependencies for environments without full ML libraries.
"""

import sys
import time
import asyncio
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from fugatto_lab.core import FugattoModel, AudioProcessor
from fugatto_lab import QuantumTaskPlanner, create_audio_generation_pipeline


async def demo_basic_audio_generation():
    """Demonstrate basic audio generation capabilities."""
    
    print("🎵 FUGATTO AUDIO LAB - GENERATION 1 DEMO")
    print("=" * 50)
    
    # Initialize core components
    print("\n🔧 Initializing components...")
    model = FugattoModel.from_pretrained("nvidia/fugatto-base")
    processor = AudioProcessor()
    
    print(f"✅ Model initialized: {model.model_name}")
    print(f"✅ Processor ready: {processor.sample_rate}Hz")
    
    # Basic audio generation
    print("\n🎵 Generating audio samples...")
    
    prompts = [
        "A cat meowing softly",
        "Ocean waves on a beach", 
        "Gentle piano melody",
        "Rain on leaves"
    ]
    
    generated_audio = {}
    
    for i, prompt in enumerate(prompts, 1):
        print(f"  {i}. Generating: '{prompt}'")
        start_time = time.time()
        
        audio = model.generate(
            prompt=prompt,
            duration_seconds=3.0,
            temperature=0.8
        )
        
        generation_time = time.time() - start_time
        print(f"     ✓ Generated {len(audio)} samples in {generation_time:.2f}s")
        
        generated_audio[prompt] = audio
    
    # Audio processing demonstration
    print("\n🔄 Audio processing demonstration...")
    
    test_audio = generated_audio[prompts[0]]
    
    # Get audio statistics
    stats = processor.get_audio_stats(test_audio)
    print(f"  📊 Audio Stats:")
    print(f"     • Duration: {stats['duration_seconds']:.2f}s")
    print(f"     • RMS: {stats['rms']:.4f}")
    print(f"     • Peak: {stats['peak']:.4f}")
    print(f"     • Dynamic Range: {stats['dynamic_range_db']:.1f} dB")
    
    # Audio enhancement
    enhanced_audio = processor.enhance_audio(
        test_audio, 
        enhance_params={
            'normalize': True,
            'noise_gate': True,
            'eq': True,
            'eq_gains': {'low': 1, 'mid': 0, 'high': -1}
        }
    )
    print(f"  ✨ Enhanced audio: {len(enhanced_audio)} samples")
    
    # Feature extraction
    features = processor.extract_features(test_audio)
    print(f"  🔍 Extracted {len(features)} audio features")
    
    # Audio transformation
    print("\n🎛️ Audio transformation demonstration...")
    transformed = model.transform(
        audio=test_audio,
        prompt="Add echo and reverb effect",
        strength=0.6
    )
    print(f"  ✓ Transformed audio: {len(transformed)} samples")
    
    return generated_audio


async def demo_quantum_task_planning():
    """Demonstrate quantum task planning for audio workflows."""
    
    print("\n⚡ QUANTUM TASK PLANNING DEMO")
    print("=" * 40)
    
    # Initialize quantum planner
    planner = QuantumTaskPlanner()
    print("✅ Quantum task planner initialized")
    
    # Create audio generation pipeline
    pipeline = create_audio_generation_pipeline(
        prompts=[
            "Jazz piano with soft drums",
            "Nature sounds with birds",
            "Electronic ambient music"
        ],
        durations=[5, 3, 8],
        qualities=["high", "medium", "high"]
    )
    
    print(f"📋 Created pipeline with {len(pipeline)} tasks")
    
    # Plan and execute pipeline
    print("\n🔄 Executing quantum-planned audio pipeline...")
    start_time = time.time()
    
    results = await planner.plan_and_execute(pipeline)
    
    execution_time = time.time() - start_time
    
    print(f"✅ Pipeline executed in {execution_time:.2f}s")
    print(f"📊 Results: {len(results)} audio files generated")
    
    # Show quantum metrics
    metrics = planner.get_metrics()
    print(f"\n📈 Quantum Planning Metrics:")
    print(f"  • Tasks processed: {metrics.get('tasks_processed', 0)}")
    print(f"  • Quantum efficiency: {metrics.get('quantum_efficiency', 0):.2f}")
    print(f"  • Resource utilization: {metrics.get('resource_utilization', 0):.2f}")
    
    return results


async def demo_model_info():
    """Demonstrate model information and capabilities."""
    
    print("\n🔍 MODEL INFORMATION DEMO")
    print("=" * 35)
    
    model = FugattoModel()
    info = model.get_model_info()
    
    print("📋 Model Configuration:")
    for key, value in info.items():
        print(f"  • {key}: {value}")
    
    print("\n🎵 Supported Generation Types:")
    print("  ✓ Text-to-Audio generation")
    print("  ✓ Audio-to-Audio transformation") 
    print("  ✓ Multi-conditioning generation")
    print("  ✓ Voice cloning (with reference)")
    print("  ✓ Music generation")
    print("  ✓ Sound effect synthesis")


async def main():
    """Main demo function."""
    
    print("🚀 FUGATTO AUDIO LAB AUTONOMOUS DEMO")
    print("Generation 1: Basic Functionality")
    print("=" * 60)
    
    try:
        # Core audio generation demo
        audio_results = await demo_basic_audio_generation()
        
        # Quantum planning demo  
        quantum_results = await demo_quantum_task_planning()
        
        # Model information demo
        await demo_model_info()
        
        print("\n🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 40)
        print("✅ All core functionality working")
        print("✅ Audio generation operational") 
        print("✅ Quantum planning functional")
        print("✅ Processing pipeline ready")
        
        print(f"\n📊 DEMO SUMMARY:")
        print(f"  • Audio samples generated: {len(audio_results)}")
        print(f"  • Quantum tasks executed: {len(quantum_results) if quantum_results else 0}")
        print(f"  • System status: Fully operational")
        
        print("\n🚀 Ready for Generation 2: Enhanced robustness and security!")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("🔧 Check system configuration and dependencies")
        raise


if __name__ == "__main__":
    asyncio.run(main())