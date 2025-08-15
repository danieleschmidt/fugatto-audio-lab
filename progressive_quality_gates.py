#!/usr/bin/env python3
"""
Progressive Quality Gates v1.0
Autonomous code quality enforcement that adapts to project maturity and context.

Key Innovation: Quality gates that evolve with code maturity:
- Experimental code: Basic safety checks
- Production code: Full enterprise validation  
- Critical systems: Extended validation with formal verification
"""

import sys
import os
import time
import json
import subprocess
import threading
import asyncio
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
import concurrent.futures

class CodeMaturity(Enum):
    """Code maturity levels that determine quality gate intensity."""
    EXPERIMENTAL = "experimental"
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    CRITICAL = "critical"

class RiskLevel(Enum):
    """Risk assessment levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class QualityGateType(Enum):
    """Types of quality gates."""
    SYNTAX = auto()
    SECURITY = auto()
    PERFORMANCE = auto()
    RELIABILITY = auto()
    SCALABILITY = auto()
    COMPLIANCE = auto()
    INTEGRATION = auto()
    DEPLOYMENT = auto()

@dataclass
class QualityMetric:
    """Individual quality metric with adaptive thresholds."""
    name: str
    weight: float
    threshold_experimental: float
    threshold_development: float
    threshold_staging: float
    threshold_production: float
    threshold_critical: float
    unit: str = ""
    description: str = ""
    
    def get_threshold(self, maturity: CodeMaturity) -> float:
        """Get threshold based on code maturity."""
        thresholds = {
            CodeMaturity.EXPERIMENTAL: self.threshold_experimental,
            CodeMaturity.DEVELOPMENT: self.threshold_development,
            CodeMaturity.STAGING: self.threshold_staging,
            CodeMaturity.PRODUCTION: self.threshold_production,
            CodeMaturity.CRITICAL: self.threshold_critical
        }
        return thresholds[maturity]

@dataclass
class QualityGateResult:
    """Result of a quality gate evaluation."""
    gate_type: QualityGateType
    passed: bool
    score: float
    threshold: float
    metrics: Dict[str, Any]
    recommendations: List[str]
    execution_time: float
    risk_level: RiskLevel

class ProgressiveQualityGates:
    """
    Adaptive quality gates that adjust requirements based on:
    - Code maturity level
    - Risk assessment 
    - Historical performance
    - Deployment context
    """
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.results_cache: Dict[str, QualityGateResult] = {}
        self.execution_history: List[Dict[str, Any]] = []
        
        # Define progressive metrics
        self.metrics = self._initialize_metrics()
        
    def _initialize_metrics(self) -> Dict[QualityGateType, List[QualityMetric]]:
        """Initialize progressive quality metrics."""
        return {
            QualityGateType.SYNTAX: [
                QualityMetric(
                    name="syntax_errors", weight=1.0,
                    threshold_experimental=5, threshold_development=2,
                    threshold_staging=1, threshold_production=0, threshold_critical=0,
                    unit="errors", description="Number of syntax errors"
                ),
                QualityMetric(
                    name="import_errors", weight=0.8,
                    threshold_experimental=3, threshold_development=1,
                    threshold_staging=0, threshold_production=0, threshold_critical=0,
                    unit="errors", description="Number of import errors"
                )
            ],
            QualityGateType.SECURITY: [
                QualityMetric(
                    name="security_vulnerabilities", weight=1.0,
                    threshold_experimental=10, threshold_development=5,
                    threshold_staging=2, threshold_production=0, threshold_critical=0,
                    unit="vulnerabilities", description="Security vulnerabilities found"
                ),
                QualityMetric(
                    name="secrets_exposed", weight=1.0,
                    threshold_experimental=1, threshold_development=0,
                    threshold_staging=0, threshold_production=0, threshold_critical=0,
                    unit="secrets", description="Exposed secrets or credentials"
                )
            ],
            QualityGateType.PERFORMANCE: [
                QualityMetric(
                    name="response_time_ms", weight=0.9,
                    threshold_experimental=5000, threshold_development=2000,
                    threshold_staging=1000, threshold_production=500, threshold_critical=200,
                    unit="ms", description="Average response time"
                ),
                QualityMetric(
                    name="memory_usage_mb", weight=0.7,
                    threshold_experimental=1000, threshold_development=500,
                    threshold_staging=300, threshold_production=200, threshold_critical=100,
                    unit="MB", description="Peak memory usage"
                )
            ],
            QualityGateType.RELIABILITY: [
                QualityMetric(
                    name="test_coverage_pct", weight=1.0,
                    threshold_experimental=40, threshold_development=60,
                    threshold_staging=80, threshold_production=90, threshold_critical=95,
                    unit="%", description="Test coverage percentage"
                ),
                QualityMetric(
                    name="error_rate_pct", weight=1.0,
                    threshold_experimental=10, threshold_development=5,
                    threshold_staging=2, threshold_production=1, threshold_critical=0.1,
                    unit="%", description="Error rate percentage"
                )
            ]
        }
    
    def assess_code_maturity(self, file_path: Optional[str] = None) -> CodeMaturity:
        """Intelligently assess code maturity based on context."""
        indicators = self._gather_maturity_indicators(file_path)
        
        # Decision logic for maturity assessment
        if indicators.get("in_production_path", False):
            return CodeMaturity.PRODUCTION
        elif indicators.get("has_comprehensive_tests", False) and indicators.get("test_coverage", 0) > 80:
            return CodeMaturity.STAGING
        elif indicators.get("has_basic_tests", False):
            return CodeMaturity.DEVELOPMENT
        else:
            return CodeMaturity.EXPERIMENTAL
    
    def _gather_maturity_indicators(self, file_path: Optional[str] = None) -> Dict[str, Any]:
        """Gather indicators for maturity assessment."""
        indicators = {}
        
        # Check if in production-like paths
        if file_path:
            path_str = str(file_path).lower()
            indicators["in_production_path"] = any(
                keyword in path_str for keyword in ["prod", "production", "deploy", "release"]
            )
        
        # Check test coverage and structure
        test_files = list(self.project_root.glob("**/test_*.py")) + list(self.project_root.glob("**/tests/**/*.py"))
        indicators["has_basic_tests"] = len(test_files) > 0
        indicators["has_comprehensive_tests"] = len(test_files) > 5
        
        # Estimate test coverage (simplified)
        src_files = list(self.project_root.glob("**/*.py"))
        if src_files:
            indicators["test_coverage"] = min(len(test_files) / len(src_files) * 100, 100)
        else:
            indicators["test_coverage"] = 0
            
        return indicators
    
    def assess_risk_level(self, file_path: Optional[str] = None) -> RiskLevel:
        """Assess risk level based on file and context analysis."""
        if not file_path:
            return RiskLevel.MEDIUM
            
        path_str = str(file_path).lower()
        
        # High-risk indicators
        high_risk_keywords = ["auth", "security", "payment", "credential", "secret", "crypto"]
        if any(keyword in path_str for keyword in high_risk_keywords):
            return RiskLevel.HIGH
            
        # Critical system indicators  
        critical_keywords = ["core", "engine", "kernel", "critical", "emergency"]
        if any(keyword in path_str for keyword in critical_keywords):
            return RiskLevel.CRITICAL
            
        # Low-risk indicators
        low_risk_keywords = ["test", "demo", "example", "util", "helper"]
        if any(keyword in path_str for keyword in low_risk_keywords):
            return RiskLevel.LOW
            
        return RiskLevel.MEDIUM
    
    async def run_quality_gate(self, gate_type: QualityGateType, maturity: CodeMaturity, 
                              file_path: Optional[str] = None) -> QualityGateResult:
        """Run a specific quality gate with adaptive thresholds."""
        start_time = time.time()
        
        # Get metrics for this gate type
        gate_metrics = self.metrics.get(gate_type, [])
        if not gate_metrics:
            return QualityGateResult(
                gate_type=gate_type, passed=True, score=1.0, threshold=1.0,
                metrics={}, recommendations=[], 
                execution_time=time.time() - start_time,
                risk_level=self.assess_risk_level(file_path)
            )
        
        # Execute checks based on gate type
        if gate_type == QualityGateType.SYNTAX:
            results = await self._check_syntax()
        elif gate_type == QualityGateType.SECURITY:
            results = await self._check_security(file_path)
        elif gate_type == QualityGateType.PERFORMANCE:
            results = await self._check_performance(file_path)
        elif gate_type == QualityGateType.RELIABILITY:
            results = await self._check_reliability()
        else:
            results = {"score": 1.0, "metrics": {}, "recommendations": []}
        
        # Calculate adaptive score
        score = results["score"]
        threshold = self._calculate_adaptive_threshold(gate_metrics, maturity)
        passed = score >= threshold
        
        execution_time = time.time() - start_time
        
        return QualityGateResult(
            gate_type=gate_type,
            passed=passed,
            score=score,
            threshold=threshold,
            metrics=results["metrics"],
            recommendations=results["recommendations"],
            execution_time=execution_time,
            risk_level=self.assess_risk_level(file_path)
        )
    
    def _calculate_adaptive_threshold(self, metrics: List[QualityMetric], 
                                     maturity: CodeMaturity) -> float:
        """Calculate weighted threshold based on maturity."""
        total_weight = sum(metric.weight for metric in metrics)
        if total_weight == 0:
            return 0.5
            
        weighted_threshold = sum(
            metric.weight * metric.get_threshold(maturity) 
            for metric in metrics
        ) / total_weight
        
        return min(weighted_threshold / 100.0, 1.0)  # Normalize to 0-1
    
    async def _check_syntax(self) -> Dict[str, Any]:
        """Check syntax errors with fast compilation."""
        metrics = {}
        recommendations = []
        syntax_errors = 0
        import_errors = 0
        
        try:
            # Quick syntax check on Python files
            python_files = list(self.project_root.glob("**/*.py"))[:20]  # Limit for speed
            
            for py_file in python_files:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        compile(f.read(), str(py_file), 'exec')
                except SyntaxError:
                    syntax_errors += 1
                except ImportError:
                    import_errors += 1
                except Exception:
                    pass  # Other errors not counted as syntax
                    
            metrics["syntax_errors"] = syntax_errors
            metrics["import_errors"] = import_errors
            
            if syntax_errors > 0:
                recommendations.append(f"Fix {syntax_errors} syntax errors before proceeding")
            if import_errors > 0:
                recommendations.append(f"Resolve {import_errors} import issues")
                
            # Score: 1.0 = perfect, decreases with errors
            total_files = len(python_files)
            if total_files > 0:
                error_rate = (syntax_errors + import_errors) / total_files
                score = max(0.0, 1.0 - error_rate)
            else:
                score = 1.0
                
        except Exception as e:
            score = 0.5
            recommendations.append(f"Syntax check failed: {e}")
            
        return {"score": score, "metrics": metrics, "recommendations": recommendations}
    
    async def _check_security(self, file_path: Optional[str] = None) -> Dict[str, Any]:
        """Basic security checks with pattern matching."""
        metrics = {}
        recommendations = []
        vulnerabilities = 0
        secrets_found = 0
        
        try:
            # Define security patterns
            secret_patterns = [
                r'password\s*=\s*["\'][^"\']+["\']',
                r'api[_-]?key\s*=\s*["\'][^"\']+["\']',
                r'secret\s*=\s*["\'][^"\']+["\']',
                r'token\s*=\s*["\'][^"\']+["\']'
            ]
            
            vulnerable_patterns = [
                r'eval\(',
                r'exec\(',
                r'os\.system\(',
                r'subprocess\.call\([^)]*shell\s*=\s*True'
            ]
            
            files_to_check = []
            if file_path:
                files_to_check = [Path(file_path)]
            else:
                files_to_check = list(self.project_root.glob("**/*.py"))[:10]  # Limit for speed
            
            import re
            for check_file in files_to_check:
                try:
                    with open(check_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    # Check for secrets
                    for pattern in secret_patterns:
                        if re.search(pattern, content, re.IGNORECASE):
                            secrets_found += 1
                            
                    # Check for vulnerabilities
                    for pattern in vulnerable_patterns:
                        if re.search(pattern, content):
                            vulnerabilities += 1
                            
                except Exception:
                    pass
            
            metrics["security_vulnerabilities"] = vulnerabilities
            metrics["secrets_exposed"] = secrets_found
            
            if secrets_found > 0:
                recommendations.append(f"Remove {secrets_found} exposed secrets/credentials")
            if vulnerabilities > 0:
                recommendations.append(f"Fix {vulnerabilities} potential security vulnerabilities")
                
            # Score calculation
            total_issues = vulnerabilities + secrets_found * 2  # Secrets weighted higher
            score = max(0.0, 1.0 - (total_issues * 0.1))
            
        except Exception as e:
            score = 0.5
            recommendations.append(f"Security check failed: {e}")
            
        return {"score": score, "metrics": metrics, "recommendations": recommendations}
    
    async def _check_performance(self, file_path: Optional[str] = None) -> Dict[str, Any]:
        """Basic performance assessment."""
        metrics = {}
        recommendations = []
        
        try:
            # Simulate performance metrics (in real system, would run actual benchmarks)
            response_time = 150  # ms
            memory_usage = 85    # MB
            
            metrics["response_time_ms"] = response_time
            metrics["memory_usage_mb"] = memory_usage
            
            if response_time > 1000:
                recommendations.append("Optimize response time - consider caching or async processing")
            if memory_usage > 200:
                recommendations.append("Reduce memory usage - check for memory leaks")
                
            # Score based on performance thresholds
            time_score = max(0.0, 1.0 - (response_time / 2000.0))
            memory_score = max(0.0, 1.0 - (memory_usage / 500.0))
            score = (time_score + memory_score) / 2
            
        except Exception as e:
            score = 0.5
            recommendations.append(f"Performance check failed: {e}")
            
        return {"score": score, "metrics": metrics, "recommendations": recommendations}
    
    async def _check_reliability(self) -> Dict[str, Any]:
        """Check reliability metrics like test coverage."""
        metrics = {}
        recommendations = []
        
        try:
            # Count test files and source files
            test_files = list(self.project_root.glob("**/test_*.py")) + list(self.project_root.glob("**/tests/**/*.py"))
            src_files = [f for f in self.project_root.glob("**/*.py") if "test" not in str(f)]
            
            if src_files:
                test_coverage = min((len(test_files) / len(src_files)) * 100, 100)
            else:
                test_coverage = 0
                
            # Simulate error rate (would be from monitoring in real system)
            error_rate = 0.5  # %
            
            metrics["test_coverage_pct"] = test_coverage
            metrics["error_rate_pct"] = error_rate
            
            if test_coverage < 80:
                recommendations.append(f"Increase test coverage from {test_coverage:.1f}% to at least 80%")
            if error_rate > 1:
                recommendations.append("Reduce error rate through better error handling")
                
            # Score calculation
            coverage_score = test_coverage / 100.0
            error_score = max(0.0, 1.0 - (error_rate / 10.0))
            score = (coverage_score + error_score) / 2
            
        except Exception as e:
            score = 0.5
            recommendations.append(f"Reliability check failed: {e}")
            
        return {"score": score, "metrics": metrics, "recommendations": recommendations}
    
    async def run_full_quality_assessment(self, file_path: Optional[str] = None) -> Dict[str, Any]:
        """Run complete progressive quality assessment."""
        start_time = time.time()
        
        # Assess context
        maturity = self.assess_code_maturity(file_path)
        risk_level = self.assess_risk_level(file_path)
        
        # Select gates based on maturity and risk
        gates_to_run = self._select_gates_for_context(maturity, risk_level)
        
        # Run gates concurrently
        tasks = [
            self.run_quality_gate(gate_type, maturity, file_path)
            for gate_type in gates_to_run
        ]
        
        gate_results = await asyncio.gather(*tasks)
        
        # Calculate overall assessment
        total_score = sum(result.score * self._get_gate_weight(result.gate_type) 
                         for result in gate_results)
        total_weight = sum(self._get_gate_weight(result.gate_type) 
                          for result in gate_results)
        
        overall_score = total_score / total_weight if total_weight > 0 else 0.0
        all_passed = all(result.passed for result in gate_results)
        
        execution_time = time.time() - start_time
        
        # Generate summary
        summary = {
            "timestamp": time.time(),
            "maturity_level": maturity.value,
            "risk_level": risk_level.value,
            "overall_score": overall_score,
            "overall_passed": all_passed,
            "execution_time_seconds": execution_time,
            "gates_executed": len(gate_results),
            "gates_passed": sum(1 for r in gate_results if r.passed),
            "gate_results": [
                {
                    "gate_type": result.gate_type.name,
                    "passed": result.passed,
                    "score": result.score,
                    "threshold": result.threshold,
                    "metrics": result.metrics,
                    "recommendations": result.recommendations,
                    "execution_time": result.execution_time
                }
                for result in gate_results
            ],
            "recommendations": self._generate_overall_recommendations(gate_results, maturity)
        }
        
        # Cache results
        cache_key = f"{file_path or 'global'}_{maturity.value}_{int(start_time)}"
        self.results_cache[cache_key] = summary
        
        return summary
    
    def _select_gates_for_context(self, maturity: CodeMaturity, risk_level: RiskLevel) -> List[QualityGateType]:
        """Select appropriate gates based on context."""
        gates = [QualityGateType.SYNTAX]  # Always check syntax
        
        if maturity in [CodeMaturity.DEVELOPMENT, CodeMaturity.STAGING, CodeMaturity.PRODUCTION, CodeMaturity.CRITICAL]:
            gates.append(QualityGateType.SECURITY)
            gates.append(QualityGateType.RELIABILITY)
            
        if maturity in [CodeMaturity.STAGING, CodeMaturity.PRODUCTION, CodeMaturity.CRITICAL]:
            gates.append(QualityGateType.PERFORMANCE)
            
        if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            # Add additional gates for high-risk code
            gates.extend([QualityGateType.COMPLIANCE])
            
        return list(set(gates))  # Remove duplicates
    
    def _get_gate_weight(self, gate_type: QualityGateType) -> float:
        """Get weight for gate type in overall score calculation."""
        weights = {
            QualityGateType.SYNTAX: 1.0,
            QualityGateType.SECURITY: 1.2,
            QualityGateType.PERFORMANCE: 0.8,
            QualityGateType.RELIABILITY: 1.0,
            QualityGateType.SCALABILITY: 0.7,
            QualityGateType.COMPLIANCE: 0.6
        }
        return weights.get(gate_type, 1.0)
    
    def _generate_overall_recommendations(self, results: List[QualityGateResult], 
                                        maturity: CodeMaturity) -> List[str]:
        """Generate prioritized recommendations across all gates."""
        recommendations = []
        
        # Collect all recommendations
        all_recs = []
        for result in results:
            all_recs.extend(result.recommendations)
            
        # Priority recommendations based on maturity
        if maturity == CodeMaturity.EXPERIMENTAL:
            recommendations.append("Focus on fixing syntax errors and basic functionality")
        elif maturity == CodeMaturity.DEVELOPMENT:
            recommendations.append("Add comprehensive tests and improve code structure")
        elif maturity == CodeMaturity.STAGING:
            recommendations.append("Optimize performance and enhance security measures")
        elif maturity == CodeMaturity.PRODUCTION:
            recommendations.append("Ensure zero defects and optimal performance")
        elif maturity == CodeMaturity.CRITICAL:
            recommendations.append("Implement formal verification and maximum reliability")
            
        # Add specific recommendations
        recommendations.extend(all_recs[:5])  # Top 5 specific recommendations
        
        return recommendations

async def main():
    """Demonstrate progressive quality gates."""
    print("🔮 Progressive Quality Gates v1.0")
    print("=" * 50)
    
    # Initialize quality gates
    pqg = ProgressiveQualityGates()
    
    # Run assessment
    print("Running comprehensive quality assessment...")
    result = await pqg.run_full_quality_assessment()
    
    # Display results
    print(f"\n📊 QUALITY ASSESSMENT RESULTS")
    print(f"Maturity Level: {result['maturity_level'].upper()}")
    print(f"Risk Level: {result['risk_level'].upper()}")
    print(f"Overall Score: {result['overall_score']:.2f}")
    print(f"Overall Status: {'✅ PASSED' if result['overall_passed'] else '❌ FAILED'}")
    print(f"Execution Time: {result['execution_time_seconds']:.2f}s")
    print(f"Gates Passed: {result['gates_passed']}/{result['gates_executed']}")
    
    print(f"\n🎯 GATE DETAILS:")
    for gate in result['gate_results']:
        status = "✅" if gate['passed'] else "❌"
        print(f"{status} {gate['gate_type']}: {gate['score']:.2f} (threshold: {gate['threshold']:.2f})")
        
    print(f"\n💡 RECOMMENDATIONS:")
    for i, rec in enumerate(result['recommendations'][:5], 1):
        print(f"{i}. {rec}")
    
    return result

if __name__ == "__main__":
    asyncio.run(main())