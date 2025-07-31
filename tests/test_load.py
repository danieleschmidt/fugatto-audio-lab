"""Load and stress testing for Fugatto Audio Lab.

This module provides comprehensive load testing capabilities to ensure
the system performs well under various load conditions.
"""

import pytest
import asyncio
import time
import concurrent.futures
from typing import List, Dict, Any, Callable
from dataclasses import dataclass
from fugatto_lab.core import FugattoModel, AudioProcessor
from fugatto_lab.monitoring import get_monitor


@dataclass
class LoadTestConfig:
    """Configuration for load testing scenarios."""
    
    concurrent_users: int = 10
    requests_per_user: int = 5
    test_duration_seconds: int = 60
    ramp_up_time_seconds: int = 10
    audio_duration_seconds: float = 5.0
    success_rate_threshold: float = 0.95
    max_response_time_ms: float = 5000.0


@dataclass
class LoadTestResult:
    """Results from load testing."""
    
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time_ms: float
    max_response_time_ms: float
    min_response_time_ms: float
    requests_per_second: float
    success_rate: float
    errors: List[str]


class LoadTester:
    """Load testing framework for audio generation."""
    
    def __init__(self, model: FugattoModel, config: LoadTestConfig):
        self.model = model
        self.config = config
        self.monitor = get_monitor(enable_detailed_logging=True)
        self.results: List[Dict[str, Any]] = []
    
    def single_request(self, user_id: int, request_id: int) -> Dict[str, Any]:
        """Execute a single audio generation request."""
        start_time = time.time()
        
        try:
            prompt = f"Load test audio {user_id}-{request_id}"
            audio = self.model.generate(
                prompt, 
                duration_seconds=self.config.audio_duration_seconds
            )
            
            end_time = time.time()
            response_time_ms = (end_time - start_time) * 1000
            
            return {
                "user_id": user_id,
                "request_id": request_id,
                "success": True,
                "response_time_ms": response_time_ms,
                "audio_length": len(audio) if audio is not None else 0,
                "error": None
            }
        
        except Exception as e:
            end_time = time.time()
            response_time_ms = (end_time - start_time) * 1000
            
            return {
                "user_id": user_id,
                "request_id": request_id,
                "success": False,
                "response_time_ms": response_time_ms,
                "audio_length": 0,
                "error": str(e)
            }
    
    def user_simulation(self, user_id: int) -> List[Dict[str, Any]]:
        """Simulate a user making multiple requests."""
        user_results = []
        
        for request_id in range(self.config.requests_per_user):
            result = self.single_request(user_id, request_id)
            user_results.append(result)
            
            # Small delay between requests
            time.sleep(0.1)
        
        return user_results
    
    def run_load_test(self) -> LoadTestResult:
        """Execute the complete load test."""
        print(f"Starting load test with {self.config.concurrent_users} users...")
        
        start_time = time.time()
        all_results = []
        
        # Use ThreadPoolExecutor for concurrent execution
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.config.concurrent_users
        ) as executor:
            # Submit all user simulations
            futures = [
                executor.submit(self.user_simulation, user_id)
                for user_id in range(self.config.concurrent_users)
            ]
            
            # Collect results
            for future in concurrent.futures.as_completed(futures):
                try:
                    user_results = future.result()
                    all_results.extend(user_results)
                except Exception as e:
                    print(f"User simulation failed: {e}")
        
        end_time = time.time()
        total_test_time = end_time - start_time
        
        # Analyze results
        total_requests = len(all_results)
        successful_requests = sum(1 for r in all_results if r["success"])
        failed_requests = total_requests - successful_requests
        
        response_times = [r["response_time_ms"] for r in all_results]
        errors = [r["error"] for r in all_results if r["error"]]
        
        return LoadTestResult(
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            avg_response_time_ms=sum(response_times) / len(response_times) if response_times else 0,
            max_response_time_ms=max(response_times) if response_times else 0,
            min_response_time_ms=min(response_times) if response_times else 0,
            requests_per_second=total_requests / total_test_time if total_test_time > 0 else 0,
            success_rate=successful_requests / total_requests if total_requests > 0 else 0,
            errors=errors
        )


class TestLoadTesting:
    """Test cases for load testing functionality."""
    
    @pytest.fixture
    def load_config(self):
        """Provide load test configuration."""
        return LoadTestConfig(
            concurrent_users=3,  # Reduced for testing
            requests_per_user=2,
            test_duration_seconds=10,
            success_rate_threshold=0.8
        )
    
    @pytest.fixture
    def load_tester(self, fugatto_model, load_config):
        """Provide load tester instance."""
        return LoadTester(fugatto_model, load_config)
    
    def test_single_request_performance(self, load_tester):
        """Test single request performance."""
        result = load_tester.single_request(0, 0)
        
        assert "success" in result
        assert "response_time_ms" in result
        assert "audio_length" in result
        
        if result["success"]:
            assert result["response_time_ms"] > 0
            assert result["audio_length"] > 0
    
    def test_user_simulation(self, load_tester):
        """Test user simulation functionality."""
        results = load_tester.user_simulation(0)
        
        assert len(results) == load_tester.config.requests_per_user
        assert all("success" in r for r in results)
        assert all("response_time_ms" in r for r in results)
    
    @pytest.mark.slow
    def test_load_test_execution(self, load_tester):
        """Test complete load test execution."""
        result = load_tester.run_load_test()
        
        assert result.total_requests > 0
        assert result.successful_requests >= 0
        assert result.failed_requests >= 0
        assert result.total_requests == result.successful_requests + result.failed_requests
        
        # Performance assertions
        assert result.success_rate >= load_tester.config.success_rate_threshold
        assert result.avg_response_time_ms <= load_tester.config.max_response_time_ms
        
        print(f"Load test results:")
        print(f"  Total requests: {result.total_requests}")
        print(f"  Success rate: {result.success_rate:.2%}")
        print(f"  Avg response time: {result.avg_response_time_ms:.2f}ms")
        print(f"  Throughput: {result.requests_per_second:.2f} req/s")


class StressTestScenarios:
    """Predefined stress testing scenarios."""
    
    @staticmethod
    def memory_stress_test() -> LoadTestConfig:
        """Configuration for memory stress testing."""
        return LoadTestConfig(
            concurrent_users=20,
            requests_per_user=10,
            audio_duration_seconds=30.0,  # Longer audio = more memory
            success_rate_threshold=0.85
        )
    
    @staticmethod
    def throughput_stress_test() -> LoadTestConfig:
        """Configuration for throughput stress testing."""
        return LoadTestConfig(
            concurrent_users=50,
            requests_per_user=20,
            audio_duration_seconds=2.0,  # Shorter audio for faster throughput
            success_rate_threshold=0.90
        )
    
    @staticmethod
    def endurance_test() -> LoadTestConfig:
        """Configuration for endurance testing."""
        return LoadTestConfig(
            concurrent_users=5,
            requests_per_user=100,
            test_duration_seconds=300,  # 5 minutes
            audio_duration_seconds=10.0,
            success_rate_threshold=0.95
        )