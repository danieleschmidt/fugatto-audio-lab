"""Command-line interface for Fugatto Audio Lab."""

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional

from .core import FugattoModel, AudioProcessor
from .monitoring import HealthChecker


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for CLI."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def health_check() -> int:
    """Perform system health check."""
    print("🔍 Fugatto Audio Lab Health Check")
    print("=" * 40)
    
    checker = HealthChecker()
    results = checker.check_all()
    
    for component, status in results.items():
        icon = "✅" if status['healthy'] else "❌"
        print(f"{icon} {component}: {status['message']}")
    
    all_healthy = all(r['healthy'] for r in results.values())
    
    if all_healthy:
        print("\n🎉 All systems operational!")
        return 0
    else:
        print("\n⚠️  Some components need attention")
        return 1


def generate_audio(prompt: str, output: str, duration: float, temperature: float) -> int:
    """Generate audio from text prompt."""
    print(f"🎵 Generating audio: '{prompt}'")
    print(f"📁 Output: {output}")
    print(f"⏱️  Duration: {duration}s")
    
    try:
        model = FugattoModel.from_pretrained("nvidia/fugatto-base")
        processor = AudioProcessor()
        
        # Generate audio
        audio_data = model.generate(
            prompt=prompt,
            duration_seconds=duration,
            temperature=temperature
        )
        
        # Save audio
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # For now, save as numpy array (placeholder)
        import numpy as np
        np.save(output_path.with_suffix('.npy'), audio_data)
        
        print("✅ Audio generation complete!")
        return 0
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        return 1


def transform_audio(input_path: str, prompt: str, output: str, strength: float) -> int:
    """Transform existing audio with text conditioning."""
    print(f"🔄 Transforming: {input_path}")
    print(f"💭 Prompt: '{prompt}'")
    print(f"📁 Output: {output}")
    
    try:
        # Load input audio (placeholder implementation)
        import numpy as np
        input_audio = np.load(input_path)
        
        model = FugattoModel.from_pretrained("nvidia/fugatto-base")
        
        # Transform audio
        transformed_audio = model.transform(
            audio=input_audio,
            prompt=prompt,
            strength=strength
        )
        
        # Save transformed audio
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path.with_suffix('.npy'), transformed_audio)
        
        print("✅ Audio transformation complete!")
        return 0
        
    except Exception as e:
        print(f"❌ Transformation failed: {e}")
        return 1


def serve_gradio(host: str = "0.0.0.0", port: int = 7860, share: bool = False) -> int:
    """Launch Gradio web interface."""
    print(f"🚀 Starting Gradio interface...")
    print(f"🌐 Host: {host}:{port}")
    
    try:
        import gradio as gr
        
        def generate_demo(prompt: str, duration: float = 10.0, temperature: float = 0.8):
            """Gradio demo function for audio generation."""
            model = FugattoModel.from_pretrained("nvidia/fugatto-base")
            audio_data = model.generate(prompt, duration, temperature)
            # Return dummy audio for demo (placeholder)
            return (48000, audio_data)
        
        # Create Gradio interface
        interface = gr.Interface(
            fn=generate_demo,
            inputs=[
                gr.Textbox(label="Audio Prompt", placeholder="A dog barking in a large hall"),
                gr.Slider(minimum=1.0, maximum=30.0, value=10.0, label="Duration (seconds)"),
                gr.Slider(minimum=0.1, maximum=1.5, value=0.8, label="Temperature")
            ],
            outputs=gr.Audio(label="Generated Audio"),
            title="🎵 Fugatto Audio Lab",
            description="Generate audio from text prompts using NVIDIA's Fugatto model",
            examples=[
                ["A cat meowing softly", 5.0, 0.7],
                ["Ocean waves crashing on rocks", 10.0, 0.8],
                ["Birds chirping in a forest", 8.0, 0.6]
            ]
        )
        
        interface.launch(server_name=host, server_port=port, share=share)
        return 0
        
    except ImportError:
        print("❌ Gradio not installed. Install with: pip install gradio")
        return 1
    except Exception as e:
        print(f"❌ Server failed to start: {e}")
        return 1


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="fugatto-lab",
        description="🎵 Fugatto Audio Lab - AI-powered audio generation and transformation"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Health check command
    health_parser = subparsers.add_parser("health", help="Perform system health check")
    
    # Generate command
    generate_parser = subparsers.add_parser("generate", help="Generate audio from text prompt")
    generate_parser.add_argument("prompt", help="Text prompt for audio generation")
    generate_parser.add_argument("-o", "--output", default="output.wav", help="Output file path")
    generate_parser.add_argument("-d", "--duration", type=float, default=10.0, help="Duration in seconds")
    generate_parser.add_argument("-t", "--temperature", type=float, default=0.8, help="Generation temperature")
    
    # Transform command
    transform_parser = subparsers.add_parser("transform", help="Transform audio with text conditioning")
    transform_parser.add_argument("input", help="Input audio file path")
    transform_parser.add_argument("prompt", help="Text prompt for transformation")
    transform_parser.add_argument("-o", "--output", default="transformed.wav", help="Output file path")
    transform_parser.add_argument("-s", "--strength", type=float, default=0.7, help="Transformation strength")
    
    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Launch Gradio web interface")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Server host")
    serve_parser.add_argument("--port", type=int, default=7860, help="Server port")
    serve_parser.add_argument("--share", action="store_true", help="Create public shareable link")
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    if args.command == "health":
        return health_check()
    elif args.command == "generate":
        return generate_audio(args.prompt, args.output, args.duration, args.temperature)
    elif args.command == "transform":
        return transform_audio(args.input, args.prompt, args.output, args.strength)
    elif args.command == "serve":
        return serve_gradio(args.host, args.port, args.share)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())