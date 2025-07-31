"""Contract testing for Fugatto Audio Lab APIs.

This module implements contract testing to ensure API consistency
and backwards compatibility for audio generation services.
"""

import pytest
import numpy as np
from typing import Dict, Any, List
from fugatto_lab.core import FugattoModel, AudioProcessor


class AudioGenerationContract:
    """Contract definition for audio generation APIs."""
    
    @staticmethod
    def validate_audio_output(audio: np.ndarray, expected_duration: float, 
                            sample_rate: int = 48000) -> bool:
        """Validate audio output meets contract requirements."""
        if not isinstance(audio, np.ndarray):
            return False
        if audio.dtype != np.float32:
            return False
        expected_samples = int(expected_duration * sample_rate)
        if abs(len(audio) - expected_samples) > sample_rate * 0.1:  # 100ms tolerance
            return False
        if np.any(np.abs(audio) > 1.0):  # Audio should be normalized
            return False
        return True
    
    @staticmethod
    def validate_model_interface(model) -> List[str]:
        """Validate model implements required interface contract."""
        required_methods = ["generate", "transform"]
        missing_methods = []
        
        for method in required_methods:
            if not hasattr(model, method):
                missing_methods.append(method)
        
        return missing_methods


class TestAudioGenerationContract:
    """Test cases for audio generation contract compliance."""
    
    def test_generate_method_contract(self, fugatto_model):
        """Test generate method adheres to contract."""
        # Contract: generate(prompt: str, duration_seconds: float) -> np.ndarray
        audio = fugatto_model.generate("test prompt", duration_seconds=2.0)
        
        assert AudioGenerationContract.validate_audio_output(audio, 2.0)
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
    
    def test_transform_method_contract(self, fugatto_model, sample_audio):
        """Test transform method adheres to contract."""
        # Contract: transform(audio: np.ndarray, prompt: str, strength: float) -> np.ndarray
        transformed = fugatto_model.transform(sample_audio, "test prompt", strength=0.5)
        
        assert isinstance(transformed, np.ndarray)
        assert transformed.shape == sample_audio.shape
        assert transformed.dtype == sample_audio.dtype
    
    def test_model_interface_contract(self, fugatto_model):
        """Test model implements required interface."""
        missing_methods = AudioGenerationContract.validate_model_interface(fugatto_model)
        assert len(missing_methods) == 0, f"Missing required methods: {missing_methods}"


class TestAudioProcessorContract:
    """Test cases for audio processor contract compliance."""
    
    def test_load_audio_contract(self, audio_processor):
        """Test load_audio method contract."""
        # Contract: load_audio(filepath: str) -> np.ndarray
        audio = audio_processor.load_audio("test.wav")
        
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert len(audio) > 0
    
    def test_save_audio_contract(self, audio_processor, sample_audio):
        """Test save_audio method contract."""
        # Contract: save_audio(audio: np.ndarray, filepath: str, sample_rate: Optional[int]) -> None
        result = audio_processor.save_audio(sample_audio, "test_output.wav", 48000)
        
        # Should return None and not raise exceptions
        assert result is None


class ContractTestReporter:
    """Generate contract testing reports."""
    
    def __init__(self):
        self.results: List[Dict[str, Any]] = []
    
    def add_result(self, contract_name: str, test_name: str, passed: bool, 
                   details: str = "") -> None:
        """Add a contract test result."""
        self.results.append({
            "contract": contract_name,
            "test": test_name,
            "passed": passed,
            "details": details,
            "timestamp": "2025-01-31T00:00:00Z"
        })
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive contract testing report."""
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r["passed"])
        
        return {
            "summary": {
                "total_contracts_tested": len(set(r["contract"] for r in self.results)),
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": total_tests - passed_tests,
                "success_rate": passed_tests / total_tests if total_tests > 0 else 0.0
            },
            "results": self.results,
            "status": "PASS" if passed_tests == total_tests else "FAIL"
        }


@pytest.fixture
def contract_reporter():
    """Provide contract test reporter."""
    return ContractTestReporter()