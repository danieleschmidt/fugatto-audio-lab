#!/usr/bin/env python3
"""Diagnostic script for Fugatto Audio Lab.

Checks system requirements, dependencies, and configuration.
Run this when experiencing issues to help diagnose problems.
"""

import os
import sys
import platform
import subprocess
from pathlib import Path
import importlib.util


def check_python():
    """Check Python version and installation."""
    print("🐍 Python Environment")
    print("-" * 30)
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Platform: {platform.platform()}")
    
    if sys.version_info < (3, 10):
        print("❌ ERROR: Python 3.10+ is required")
        return False
    else:
        print("✅ Python version OK")
        return True


def check_cuda():
    """Check CUDA installation and availability."""
    print("\n🔥 CUDA Environment")
    print("-" * 30)
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        print(f"CUDA available: {cuda_available}")
        
        if cuda_available:
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                gpu = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {gpu.name} ({gpu.total_memory // 1024**2} MB)")
        else:
            print("⚠️  WARNING: CUDA not available, using CPU")
        
        return True
    except ImportError:
        print("❌ ERROR: PyTorch not installed")
        return False
    except Exception as e:
        print(f"❌ ERROR: CUDA check failed: {e}")
        return False


def check_dependencies():
    """Check core dependencies."""
    print("\n📦 Dependencies")
    print("-" * 30)
    
    required_packages = [
        "torch", "torchaudio", "transformers", "accelerate",
        "librosa", "soundfile", "gradio", "streamlit", "numpy"
    ]
    
    all_good = True
    for package in required_packages:
        try:
            spec = importlib.util.find_spec(package)
            if spec is not None:
                module = importlib.import_module(package)
                version = getattr(module, "__version__", "unknown")
                print(f"✅ {package}: {version}")
            else:
                print(f"❌ {package}: Not found")
                all_good = False
        except Exception as e:
            print(f"❌ {package}: Error - {e}")
            all_good = False
    
    return all_good


def check_fugatto_lab():
    """Check Fugatto Lab installation."""
    print("\n🎵 Fugatto Audio Lab")
    print("-" * 30)
    
    try:
        import fugatto_lab
        print(f"✅ Package installed: {fugatto_lab.__version__}")
        
        # Check core classes
        from fugatto_lab import FugattoModel, AudioProcessor
        print("✅ Core classes importable")
        
        # Test basic functionality
        processor = AudioProcessor()
        print("✅ Basic instantiation works")
        
        return True
    except ImportError as e:
        print(f"❌ Fugatto Lab not properly installed: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing Fugatto Lab: {e}")
        return False


def check_file_structure():
    """Check project file structure."""
    print("\n📁 File Structure")
    print("-" * 30)
    
    required_files = [
        "pyproject.toml", "README.md", "LICENSE", 
        "fugatto_lab/__init__.py", "tests/"
    ]
    
    optional_files = [
        ".env", "configs/", "models/", "outputs/", "cache/"
    ]
    
    all_good = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} (required)")
            all_good = False
    
    print("\nOptional files:")
    for file_path in optional_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"⚠️  {file_path} (optional)")
    
    return all_good


def check_system_resources():
    """Check system resources."""
    print("\n💻 System Resources")
    print("-" * 30)
    
    import psutil
    
    # Memory
    memory = psutil.virtual_memory()
    print(f"RAM: {memory.total // 1024**3} GB total, {memory.available // 1024**3} GB available")
    if memory.total < 8 * 1024**3:  # 8GB
        print("⚠️  WARNING: Less than 8GB RAM, may affect performance")
    
    # Disk space
    disk = psutil.disk_usage('.')
    print(f"Disk: {disk.total // 1024**3} GB total, {disk.free // 1024**3} GB free")
    if disk.free < 10 * 1024**3:  # 10GB
        print("⚠️  WARNING: Less than 10GB free disk space")
    
    # CPU
    print(f"CPU: {psutil.cpu_count()} cores")
    
    return True


def check_permissions():
    """Check file permissions."""
    print("\n🔐 Permissions")
    print("-" * 30)
    
    test_dirs = [".", "fugatto_lab", "tests"]
    all_good = True
    
    for directory in test_dirs:
        if Path(directory).exists():
            if os.access(directory, os.R_OK):
                print(f"✅ Read access to {directory}")
            else:
                print(f"❌ No read access to {directory}")
                all_good = False
            
            if os.access(directory, os.W_OK):
                print(f"✅ Write access to {directory}")
            else:
                print(f"❌ No write access to {directory}")
                all_good = False
    
    return all_good


def run_quick_test():
    """Run a quick functionality test."""
    print("\n🧪 Quick Test")
    print("-" * 30)
    
    try:
        from fugatto_lab import FugattoModel, AudioProcessor
        
        # Test model creation
        model = FugattoModel.from_pretrained("test-model")
        print("✅ Model creation works")
        
        # Test audio processing
        processor = AudioProcessor()
        print("✅ Audio processor works")
        
        # Test basic generation (mock)
        audio = model.generate("test prompt", duration_seconds=1.0)
        print(f"✅ Basic generation works (output shape: {audio.shape})")
        
        return True
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False


def print_recommendations():
    """Print recommendations based on checks."""
    print("\n💡 Recommendations")
    print("-" * 30)
    print("If you encountered errors:")
    print("1. Install missing dependencies: pip install -e .[dev]")
    print("2. Check CUDA installation if using GPU")
    print("3. Ensure proper file permissions")
    print("4. Create missing directories: mkdir -p logs outputs cache models")
    print("5. Copy .env.example to .env if environment file missing")
    print("\nFor more help:")
    print("- Check docs/DEVELOPMENT.md")
    print("- Run 'make help' for available commands")
    print("- Join our Discord community")


def main():
    """Run all diagnostic checks."""
    print("Fugatto Audio Lab Diagnostic Tool")
    print("=" * 50)
    
    checks = [
        ("Python", check_python),
        ("CUDA", check_cuda),
        ("Dependencies", check_dependencies),
        ("Fugatto Lab", check_fugatto_lab),
        ("File Structure", check_file_structure),
        ("System Resources", check_system_resources),
        ("Permissions", check_permissions),
        ("Quick Test", run_quick_test),
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"❌ {name} check failed with exception: {e}")
            results[name] = False
    
    # Summary
    print("\n📊 Summary")
    print("-" * 30)
    passed = sum(results.values())
    total = len(results)
    
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name}: {status}")
    
    print(f"\nOverall: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 All checks passed! System looks good.")
    else:
        print("⚠️  Some issues found. See recommendations below.")
        print_recommendations()
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)