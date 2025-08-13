#!/usr/bin/env python3
"""Autonomous Quality Verification System.

Comprehensive quality gate validation without external dependencies.
"""

import ast
import gc
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class QualityGateValidator:
    """Comprehensive quality gate validation system."""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.results = {
            'code_quality': {},
            'security_scan': {},
            'performance_check': {},
            'integration_test': {},
            'documentation_check': {},
            'dependency_audit': {}
        }
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        
    def run_all_quality_gates(self) -> Dict[str, Any]:
        """Execute all quality gate validations."""
        print("🚀 AUTONOMOUS QUALITY VERIFICATION SYSTEM")
        print("=" * 60)
        
        # 1. Code Quality Analysis
        print("\n🔍 CODE QUALITY ANALYSIS")
        self.results['code_quality'] = self.validate_code_quality()
        
        # 2. Security Scanning
        print("\n🛡️ SECURITY SCANNING")
        self.results['security_scan'] = self.validate_security()
        
        # 3. Performance Verification
        print("\n⚡ PERFORMANCE VERIFICATION")
        self.results['performance_check'] = self.validate_performance()
        
        # 4. Integration Testing
        print("\n🔗 INTEGRATION TESTING")
        self.results['integration_test'] = self.validate_integration()
        
        # 5. Documentation Validation
        print("\n📚 DOCUMENTATION VALIDATION")
        self.results['documentation_check'] = self.validate_documentation()
        
        # 6. Dependency Security Audit
        print("\n🔒 DEPENDENCY SECURITY AUDIT")
        self.results['dependency_audit'] = self.validate_dependencies()
        
        # Generate final report
        print("\n" + "=" * 60)
        return self.generate_final_report()
    
    def validate_code_quality(self) -> Dict[str, Any]:
        """Validate code quality metrics."""
        results = {
            'syntax_check': self.check_python_syntax(),
            'import_analysis': self.analyze_imports(),
            'complexity_analysis': self.analyze_complexity(),
            'code_structure': self.analyze_code_structure(),
            'naming_conventions': self.check_naming_conventions()
        }
        
        passed = sum(1 for result in results.values() if result.get('passed', False))
        total = len(results)
        
        print(f"  Code Quality: {passed}/{total} checks passed")
        return results
    
    def check_python_syntax(self) -> Dict[str, Any]:
        """Check Python syntax for all .py files."""
        python_files = list(self.project_root.glob('**/*.py'))
        syntax_errors = []
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    ast.parse(f.read())
            except SyntaxError as e:
                syntax_errors.append({
                    'file': str(py_file),
                    'line': e.lineno,
                    'error': str(e)
                })
            except Exception as e:
                syntax_errors.append({
                    'file': str(py_file),
                    'error': f"Parse error: {e}"
                })
        
        passed = len(syntax_errors) == 0
        self.total_tests += 1
        if passed:
            self.passed_tests += 1
        else:
            self.failed_tests += 1
            
        return {
            'passed': passed,
            'files_checked': len(python_files),
            'syntax_errors': syntax_errors,
            'message': f"Syntax check: {len(python_files)} files, {len(syntax_errors)} errors"
        }
    
    def analyze_imports(self) -> Dict[str, Any]:
        """Analyze import structure and dependencies."""
        python_files = list(self.project_root.glob('**/*.py'))
        import_analysis = {
            'total_imports': 0,
            'stdlib_imports': 0,
            'third_party_imports': 0,
            'local_imports': 0,
            'circular_imports': [],
            'unused_imports': []
        }
        
        stdlib_modules = {
            'os', 'sys', 'time', 'datetime', 'json', 'logging', 'threading',
            'asyncio', 'collections', 'dataclasses', 'enum', 'typing',
            'pathlib', 'hashlib', 'hmac', 'secrets', 'warnings', 'gc',
            'signal', 'random', 'math', 'ast', 'traceback'
        }
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read())
                
                for node in ast.walk(tree):
                    if isinstance(node, (ast.Import, ast.ImportFrom)):
                        import_analysis['total_imports'] += 1
                        
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                module_name = alias.name.split('.')[0]
                                if module_name in stdlib_modules:
                                    import_analysis['stdlib_imports'] += 1
                                elif module_name.startswith('fugatto_lab'):
                                    import_analysis['local_imports'] += 1
                                else:
                                    import_analysis['third_party_imports'] += 1
                        
                        elif isinstance(node, ast.ImportFrom):
                            if node.module:
                                module_name = node.module.split('.')[0]
                                if module_name in stdlib_modules:
                                    import_analysis['stdlib_imports'] += 1
                                elif module_name.startswith('fugatto_lab'):
                                    import_analysis['local_imports'] += 1
                                else:
                                    import_analysis['third_party_imports'] += 1
                            
            except Exception:
                continue
        
        passed = import_analysis['total_imports'] > 0
        self.total_tests += 1
        if passed:
            self.passed_tests += 1
        else:
            self.failed_tests += 1
            
        return {
            'passed': passed,
            'analysis': import_analysis,
            'message': f"Import analysis: {import_analysis['total_imports']} total imports"
        }
    
    def analyze_complexity(self) -> Dict[str, Any]:
        """Analyze code complexity metrics."""
        python_files = list(self.project_root.glob('**/*.py'))
        complexity_data = {
            'total_functions': 0,
            'total_classes': 0,
            'total_lines': 0,
            'high_complexity_functions': [],
            'large_functions': []
        }
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                    complexity_data['total_lines'] += len(lines)
                    
                    tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        complexity_data['total_functions'] += 1
                        
                        # Calculate function length
                        func_lines = node.end_lineno - node.lineno if hasattr(node, 'end_lineno') else 0
                        if func_lines > 50:  # Functions longer than 50 lines
                            complexity_data['large_functions'].append({
                                'file': str(py_file),
                                'function': node.name,
                                'lines': func_lines
                            })
                    
                    elif isinstance(node, ast.ClassDef):
                        complexity_data['total_classes'] += 1
                        
            except Exception:
                continue
        
        passed = (
            complexity_data['total_functions'] > 0 and
            len(complexity_data['large_functions']) < complexity_data['total_functions'] * 0.1
        )
        
        self.total_tests += 1
        if passed:
            self.passed_tests += 1
        else:
            self.failed_tests += 1
            
        return {
            'passed': passed,
            'metrics': complexity_data,
            'message': f"Complexity: {complexity_data['total_functions']} functions, {len(complexity_data['large_functions'])} large"
        }
    
    def analyze_code_structure(self) -> Dict[str, Any]:
        """Analyze overall code structure and organization."""
        structure = {
            'total_files': 0,
            'python_files': 0,
            'test_files': 0,
            'config_files': 0,
            'documentation_files': 0,
            'has_init_files': False,
            'package_structure': []
        }
        
        for file_path in self.project_root.rglob('*'):
            if file_path.is_file():
                structure['total_files'] += 1
                
                if file_path.suffix == '.py':
                    structure['python_files'] += 1
                    
                    if 'test' in file_path.name.lower():
                        structure['test_files'] += 1
                    
                    if file_path.name == '__init__.py':
                        structure['has_init_files'] = True
                        structure['package_structure'].append(str(file_path.parent))
                
                elif file_path.suffix in ['.md', '.rst', '.txt']:
                    structure['documentation_files'] += 1
                
                elif file_path.suffix in ['.toml', '.yaml', '.yml', '.json', '.ini']:
                    structure['config_files'] += 1
        
        passed = (
            structure['python_files'] > 5 and
            structure['has_init_files'] and
            structure['documentation_files'] > 0
        )
        
        self.total_tests += 1
        if passed:
            self.passed_tests += 1
        else:
            self.failed_tests += 1
            
        return {
            'passed': passed,
            'structure': structure,
            'message': f"Structure: {structure['python_files']} Python files, {structure['documentation_files']} docs"
        }
    
    def check_naming_conventions(self) -> Dict[str, Any]:
        """Check Python naming conventions."""
        python_files = list(self.project_root.glob('**/*.py'))
        naming_issues = []
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read())
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        # Classes should be PascalCase
                        if not node.name[0].isupper():
                            naming_issues.append({
                                'file': str(py_file),
                                'type': 'class',
                                'name': node.name,
                                'issue': 'Should start with uppercase'
                            })
                    
                    elif isinstance(node, ast.FunctionDef):
                        # Functions should be snake_case
                        if node.name.startswith('_'):
                            continue  # Skip private functions
                        
                        if any(c.isupper() for c in node.name) and '_' not in node.name:
                            naming_issues.append({
                                'file': str(py_file),
                                'type': 'function',
                                'name': node.name,
                                'issue': 'Should use snake_case'
                            })
                            
            except Exception:
                continue
        
        passed = len(naming_issues) < 10  # Allow some naming flexibility
        self.total_tests += 1
        if passed:
            self.passed_tests += 1
        else:
            self.failed_tests += 1
            
        return {
            'passed': passed,
            'issues': naming_issues,
            'message': f"Naming: {len(naming_issues)} convention issues found"
        }
    
    def validate_security(self) -> Dict[str, Any]:
        """Validate security aspects of the code."""
        results = {
            'hardcoded_secrets': self.check_hardcoded_secrets(),
            'sql_injection_risk': self.check_sql_injection(),
            'unsafe_functions': self.check_unsafe_functions(),
            'file_permissions': self.check_file_permissions()
        }
        
        passed = sum(1 for result in results.values() if result.get('passed', False))
        total = len(results)
        
        print(f"  Security: {passed}/{total} checks passed")
        return results
    
    def check_hardcoded_secrets(self) -> Dict[str, Any]:
        """Check for hardcoded secrets and credentials."""
        python_files = list(self.project_root.glob('**/*.py'))
        secret_patterns = [
            'password', 'secret', 'key', 'token', 'api_key',
            'private_key', 'credential', 'auth'
        ]
        
        potential_secrets = []
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                    lines = f.readlines()
                
                for i, line in enumerate(lines):
                    line_lower = line.lower()
                    # Check for assignment patterns
                    if '=' in line and any(pattern in line_lower for pattern in secret_patterns):
                        # Skip comments and docstrings
                        if line.strip().startswith('#') or '"""' in line or "'''" in line:
                            continue
                        
                        # Skip obvious test/example values
                        if any(test_val in line_lower for test_val in ['test', 'example', 'demo', 'mock', 'fake']):
                            continue
                        
                        potential_secrets.append({
                            'file': str(py_file),
                            'line': i + 1,
                            'content': line.strip()[:100]  # First 100 chars
                        })
                        
            except Exception:
                continue
        
        # Filter out obvious false positives
        filtered_secrets = []
        for secret in potential_secrets:
            content = secret['content'].lower()
            if not any(safe_pattern in content for safe_pattern in [
                'none', 'null', 'false', 'true', '""', "''", 'optional', 'default'
            ]):
                filtered_secrets.append(secret)
        
        passed = len(filtered_secrets) == 0
        self.total_tests += 1
        if passed:
            self.passed_tests += 1
        else:
            self.failed_tests += 1
            
        return {
            'passed': passed,
            'potential_secrets': filtered_secrets,
            'message': f"Hardcoded secrets: {len(filtered_secrets)} potential issues found"
        }
    
    def check_sql_injection(self) -> Dict[str, Any]:
        """Check for SQL injection vulnerabilities."""
        python_files = list(self.project_root.glob('**/*.py'))
        sql_risks = []
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                for i, line in enumerate(lines):
                    line_lower = line.lower()
                    # Check for string formatting in SQL-like statements
                    if ('select' in line_lower or 'insert' in line_lower or 
                        'update' in line_lower or 'delete' in line_lower):
                        if '%' in line or '.format(' in line or 'f"' in line or "f'" in line:
                            sql_risks.append({
                                'file': str(py_file),
                                'line': i + 1,
                                'content': line.strip()[:100]
                            })
                            
            except Exception:
                continue
        
        passed = len(sql_risks) == 0
        self.total_tests += 1
        if passed:
            self.passed_tests += 1
        else:
            self.failed_tests += 1
            
        return {
            'passed': passed,
            'risks': sql_risks,
            'message': f"SQL injection: {len(sql_risks)} potential risks found"
        }
    
    def check_unsafe_functions(self) -> Dict[str, Any]:
        """Check for usage of unsafe functions."""
        python_files = list(self.project_root.glob('**/*.py'))
        unsafe_functions = ['eval', 'exec', 'compile', '__import__']
        unsafe_usage = []
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read())
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call):
                        if isinstance(node.func, ast.Name):
                            if node.func.id in unsafe_functions:
                                unsafe_usage.append({
                                    'file': str(py_file),
                                    'function': node.func.id,
                                    'line': getattr(node, 'lineno', 0)
                                })
                                
            except Exception:
                continue
        
        passed = len(unsafe_usage) == 0
        self.total_tests += 1
        if passed:
            self.passed_tests += 1
        else:
            self.failed_tests += 1
            
        return {
            'passed': passed,
            'unsafe_usage': unsafe_usage,
            'message': f"Unsafe functions: {len(unsafe_usage)} usages found"
        }
    
    def check_file_permissions(self) -> Dict[str, Any]:
        """Check file permissions for security issues."""
        sensitive_files = []
        
        for file_path in self.project_root.rglob('*'):
            if file_path.is_file():
                try:
                    # Check if file is executable when it shouldn't be
                    if file_path.suffix in ['.py', '.md', '.txt', '.json', '.yaml', '.yml']:
                        if os.access(file_path, os.X_OK) and file_path.name != '__main__.py':
                            sensitive_files.append({
                                'file': str(file_path),
                                'issue': 'Executable permission on non-executable file'
                            })
                except Exception:
                    continue
        
        passed = len(sensitive_files) == 0
        self.total_tests += 1
        if passed:
            self.passed_tests += 1
        else:
            self.failed_tests += 1
            
        return {
            'passed': passed,
            'issues': sensitive_files,
            'message': f"File permissions: {len(sensitive_files)} issues found"
        }
    
    def validate_performance(self) -> Dict[str, Any]:
        """Validate performance characteristics."""
        results = {
            'memory_usage': self.check_memory_usage(),
            'import_performance': self.check_import_performance(),
            'function_performance': self.check_function_performance()
        }
        
        passed = sum(1 for result in results.values() if result.get('passed', False))
        total = len(results)
        
        print(f"  Performance: {passed}/{total} checks passed")
        return results
    
    def check_memory_usage(self) -> Dict[str, Any]:
        """Check memory usage patterns."""
        # Basic memory check
        initial_memory = self.get_memory_usage()
        
        # Try importing main modules
        try:
            sys.path.insert(0, str(self.project_root))
            import fugatto_lab
            post_import_memory = self.get_memory_usage()
            
            memory_increase = post_import_memory - initial_memory
            passed = memory_increase < 100  # Less than 100MB increase
            
        except Exception as e:
            memory_increase = 0
            passed = False
        
        self.total_tests += 1
        if passed:
            self.passed_tests += 1
        else:
            self.failed_tests += 1
            
        return {
            'passed': passed,
            'initial_memory_mb': initial_memory,
            'memory_increase_mb': memory_increase,
            'message': f"Memory: {memory_increase:.1f}MB increase on import"
        }
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            # Fallback: use gc to estimate
            objects = len(gc.get_objects())
            return objects * 0.001  # Rough estimate
    
    def check_import_performance(self) -> Dict[str, Any]:
        """Check import performance."""
        import_times = {}
        
        modules_to_test = [
            'fugatto_lab.quantum_planner',
            'fugatto_lab.core',
            'fugatto_lab.adaptive_quantum_streaming',
            'fugatto_lab.neural_adaptive_enhancement'
        ]
        
        sys.path.insert(0, str(self.project_root))
        
        for module_name in modules_to_test:
            try:
                start_time = time.time()
                __import__(module_name)
                import_time = time.time() - start_time
                import_times[module_name] = import_time
            except Exception as e:
                import_times[module_name] = f"Failed: {e}"
        
        # Check if any imports are too slow (>5 seconds)
        slow_imports = {k: v for k, v in import_times.items() 
                       if isinstance(v, float) and v > 5.0}
        
        passed = len(slow_imports) == 0
        self.total_tests += 1
        if passed:
            self.passed_tests += 1
        else:
            self.failed_tests += 1
            
        return {
            'passed': passed,
            'import_times': import_times,
            'slow_imports': slow_imports,
            'message': f"Import performance: {len(slow_imports)} slow imports"
        }
    
    def check_function_performance(self) -> Dict[str, Any]:
        """Check basic function performance."""
        try:
            sys.path.insert(0, str(self.project_root))
            from fugatto_lab.quantum_planner import QuantumTask, TaskPriority
            
            # Test basic operations
            start_time = time.time()
            
            # Create many tasks
            tasks = []
            for i in range(1000):
                task = QuantumTask(
                    task_id=f"perf_test_{i}",
                    task_type="test",
                    priority=TaskPriority.MEDIUM,
                    estimated_duration=1.0,
                    dependencies=[],
                    parameters={"test_param": i}
                )
                tasks.append(task)
            
            creation_time = time.time() - start_time
            passed = creation_time < 1.0  # Should create 1000 tasks in <1 second
            
        except Exception as e:
            creation_time = 0
            passed = False
        
        self.total_tests += 1
        if passed:
            self.passed_tests += 1
        else:
            self.failed_tests += 1
            
        return {
            'passed': passed,
            'task_creation_time': creation_time,
            'message': f"Function performance: {creation_time:.3f}s for 1000 tasks"
        }
    
    def validate_integration(self) -> Dict[str, Any]:
        """Validate integration between components."""
        results = {
            'module_integration': self.test_module_integration(),
            'api_integration': self.test_api_integration(),
            'service_integration': self.test_service_integration()
        }
        
        passed = sum(1 for result in results.values() if result.get('passed', False))
        total = len(results)
        
        print(f"  Integration: {passed}/{total} checks passed")
        return results
    
    def test_module_integration(self) -> Dict[str, Any]:
        """Test integration between modules."""
        try:
            sys.path.insert(0, str(self.project_root))
            
            # Test quantum planner integration
            from fugatto_lab import QuantumTaskPlanner, create_audio_generation_pipeline
            
            planner = QuantumTaskPlanner()
            pipeline = create_audio_generation_pipeline()
            
            # Test basic integration
            planner.add_task_pipeline(pipeline)
            
            passed = True
            message = "Module integration successful"
            
        except Exception as e:
            passed = False
            message = f"Module integration failed: {e}"
        
        self.total_tests += 1
        if passed:
            self.passed_tests += 1
        else:
            self.failed_tests += 1
            
        return {
            'passed': passed,
            'message': message
        }
    
    def test_api_integration(self) -> Dict[str, Any]:
        """Test API integration capabilities."""
        try:
            sys.path.insert(0, str(self.project_root))
            from fugatto_lab.api.app import create_app
            
            # Test app creation
            app = create_app({'debug': True, 'enable_docs': False})
            
            passed = app is not None
            message = "API integration successful" if passed else "API integration failed"
            
        except Exception as e:
            passed = False
            message = f"API integration failed: {e}"
        
        self.total_tests += 1
        if passed:
            self.passed_tests += 1
        else:
            self.failed_tests += 1
            
        return {
            'passed': passed,
            'message': message
        }
    
    def test_service_integration(self) -> Dict[str, Any]:
        """Test service integration."""
        try:
            sys.path.insert(0, str(self.project_root))
            from fugatto_lab.core import FugattoModel, AudioProcessor
            
            # Test service creation
            model = FugattoModel()
            processor = AudioProcessor()
            
            # Test basic interaction
            info = model.get_model_info()
            
            passed = info is not None and isinstance(info, dict)
            message = "Service integration successful"
            
        except Exception as e:
            passed = False
            message = f"Service integration failed: {e}"
        
        self.total_tests += 1
        if passed:
            self.passed_tests += 1
        else:
            self.failed_tests += 1
            
        return {
            'passed': passed,
            'message': message
        }
    
    def validate_documentation(self) -> Dict[str, Any]:
        """Validate documentation completeness."""
        results = {
            'readme_check': self.check_readme(),
            'docstring_coverage': self.check_docstring_coverage(),
            'api_documentation': self.check_api_docs(),
            'example_code': self.check_example_code()
        }
        
        passed = sum(1 for result in results.values() if result.get('passed', False))
        total = len(results)
        
        print(f"  Documentation: {passed}/{total} checks passed")
        return results
    
    def check_readme(self) -> Dict[str, Any]:
        """Check README file completeness."""
        readme_files = list(self.project_root.glob('README*'))
        
        if not readme_files:
            passed = False
            message = "No README file found"
        else:
            readme_file = readme_files[0]
            try:
                with open(readme_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                required_sections = [
                    'installation', 'usage', 'example', 'feature',
                    'requirement', 'quick start', 'overview'
                ]
                
                content_lower = content.lower()
                found_sections = sum(1 for section in required_sections 
                                   if section in content_lower)
                
                passed = found_sections >= 3 and len(content) > 1000
                message = f"README: {found_sections}/{len(required_sections)} sections, {len(content)} chars"
                
            except Exception as e:
                passed = False
                message = f"README read error: {e}"
        
        self.total_tests += 1
        if passed:
            self.passed_tests += 1
        else:
            self.failed_tests += 1
            
        return {
            'passed': passed,
            'message': message
        }
    
    def check_docstring_coverage(self) -> Dict[str, Any]:
        """Check docstring coverage for functions and classes."""
        python_files = list(self.project_root.glob('**/*.py'))
        total_functions = 0
        documented_functions = 0
        total_classes = 0
        documented_classes = 0
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read())
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        total_functions += 1
                        if ast.get_docstring(node):
                            documented_functions += 1
                    
                    elif isinstance(node, ast.ClassDef):
                        total_classes += 1
                        if ast.get_docstring(node):
                            documented_classes += 1
                            
            except Exception:
                continue
        
        function_coverage = documented_functions / max(1, total_functions)
        class_coverage = documented_classes / max(1, total_classes)
        overall_coverage = (documented_functions + documented_classes) / max(1, total_functions + total_classes)
        
        passed = overall_coverage >= 0.6  # 60% coverage requirement
        
        self.total_tests += 1
        if passed:
            self.passed_tests += 1
        else:
            self.failed_tests += 1
            
        return {
            'passed': passed,
            'function_coverage': function_coverage,
            'class_coverage': class_coverage,
            'overall_coverage': overall_coverage,
            'message': f"Docstring coverage: {overall_coverage:.1%} overall"
        }
    
    def check_api_docs(self) -> Dict[str, Any]:
        """Check API documentation availability."""
        api_files = list(self.project_root.glob('**/api/**/*.py'))
        documented_endpoints = 0
        total_endpoints = 0
        
        for api_file in api_files:
            try:
                with open(api_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Look for route decorators
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if '@app.route' in line or '@router.' in line or 'def ' in line:
                        if 'def ' in line and not line.strip().startswith('#'):
                            total_endpoints += 1
                            
                            # Check for docstring in next few lines
                            for j in range(i + 1, min(i + 5, len(lines))):
                                if '"""' in lines[j] or "'''" in lines[j]:
                                    documented_endpoints += 1
                                    break
                                if 'def ' in lines[j]:  # Found another function
                                    break
                                    
            except Exception:
                continue
        
        if total_endpoints == 0:
            passed = True  # No API endpoints to document
            coverage = 1.0
        else:
            coverage = documented_endpoints / total_endpoints
            passed = coverage >= 0.7
        
        self.total_tests += 1
        if passed:
            self.passed_tests += 1
        else:
            self.failed_tests += 1
            
        return {
            'passed': passed,
            'documented_endpoints': documented_endpoints,
            'total_endpoints': total_endpoints,
            'coverage': coverage,
            'message': f"API docs: {documented_endpoints}/{total_endpoints} endpoints documented"
        }
    
    def check_example_code(self) -> Dict[str, Any]:
        """Check for example code availability."""
        example_indicators = [
            'example', 'demo', 'sample', 'tutorial', 'quickstart',
            'if __name__ == "__main__"', 'main()', 'demonstrate'
        ]
        
        python_files = list(self.project_root.glob('**/*.py'))
        files_with_examples = 0
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                
                if any(indicator in content for indicator in example_indicators):
                    files_with_examples += 1
                    
            except Exception:
                continue
        
        passed = files_with_examples >= 3  # At least 3 files with examples
        
        self.total_tests += 1
        if passed:
            self.passed_tests += 1
        else:
            self.failed_tests += 1
            
        return {
            'passed': passed,
            'files_with_examples': files_with_examples,
            'total_files': len(python_files),
            'message': f"Examples: {files_with_examples} files with example code"
        }
    
    def validate_dependencies(self) -> Dict[str, Any]:
        """Validate dependency security and compatibility."""
        results = {
            'dependency_analysis': self.analyze_dependencies(),
            'version_compatibility': self.check_version_compatibility(),
            'circular_dependencies': self.check_circular_dependencies()
        }
        
        passed = sum(1 for result in results.values() if result.get('passed', False))
        total = len(results)
        
        print(f"  Dependencies: {passed}/{total} checks passed")
        return results
    
    def analyze_dependencies(self) -> Dict[str, Any]:
        """Analyze project dependencies."""
        requirements_files = list(self.project_root.glob('requirements*.txt'))
        pyproject_file = self.project_root / 'pyproject.toml'
        
        dependencies = set()
        optional_dependencies = set()
        
        # Check requirements.txt files
        for req_file in requirements_files:
            try:
                with open(req_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            # Extract package name
                            pkg_name = line.split('>=')[0].split('==')[0].split('<')[0].split('>')[0]
                            dependencies.add(pkg_name)
            except Exception:
                continue
        
        # Check pyproject.toml
        if pyproject_file.exists():
            try:
                with open(pyproject_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Simple parsing for dependencies
                in_dependencies = False
                for line in content.split('\n'):
                    if 'dependencies = [' in line:
                        in_dependencies = True
                        continue
                    elif in_dependencies and ']' in line:
                        in_dependencies = False
                        continue
                    elif in_dependencies and '"' in line:
                        # Extract package name from quoted string
                        pkg = line.split('"')[1].split('>=')[0].split('==')[0]
                        dependencies.add(pkg)
                        
            except Exception:
                pass
        
        # Check for security-sensitive dependencies
        sensitive_deps = {'cryptography', 'pycryptodome', 'jwt', 'oauth'}
        security_deps = dependencies.intersection(sensitive_deps)
        
        passed = len(dependencies) > 0 and len(dependencies) < 50  # Reasonable number
        
        self.total_tests += 1
        if passed:
            self.passed_tests += 1
        else:
            self.failed_tests += 1
            
        return {
            'passed': passed,
            'total_dependencies': len(dependencies),
            'security_related': list(security_deps),
            'dependencies': list(dependencies),
            'message': f"Dependencies: {len(dependencies)} total, {len(security_deps)} security-related"
        }
    
    def check_version_compatibility(self) -> Dict[str, Any]:
        """Check Python version compatibility."""
        python_files = list(self.project_root.glob('**/*.py'))
        compatibility_issues = []
        
        # Check for Python 3.10+ features
        modern_features = [
            'match ', 'case ', '|', 'Union[', 'Optional[',
            'typing.', 'dataclasses.', 'pathlib.'
        ]
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for modern syntax
                uses_modern = any(feature in content for feature in modern_features)
                
                if uses_modern:
                    # Check if file specifies Python version requirement
                    if 'python_requires' not in content and '>=3.10' not in content:
                        compatibility_issues.append({
                            'file': str(py_file),
                            'issue': 'Uses modern Python features without version specification'
                        })
                        
            except Exception:
                continue
        
        passed = len(compatibility_issues) < 5  # Allow some flexibility
        
        self.total_tests += 1
        if passed:
            self.passed_tests += 1
        else:
            self.failed_tests += 1
            
        return {
            'passed': passed,
            'issues': compatibility_issues,
            'message': f"Version compatibility: {len(compatibility_issues)} issues found"
        }
    
    def check_circular_dependencies(self) -> Dict[str, Any]:
        """Check for circular import dependencies."""
        python_files = list(self.project_root.glob('**/*.py'))
        import_graph = {}
        
        # Build import graph
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read())
                
                module_name = str(py_file.relative_to(self.project_root)).replace('/', '.').replace('.py', '')
                imports = set()
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom):
                        if node.module and node.module.startswith('fugatto_lab'):
                            imports.add(node.module)
                    elif isinstance(node, ast.Import):
                        for alias in node.names:
                            if alias.name.startswith('fugatto_lab'):
                                imports.add(alias.name)
                
                import_graph[module_name] = imports
                
            except Exception:
                continue
        
        # Simple circular dependency detection
        circular_deps = []
        for module, deps in import_graph.items():
            for dep in deps:
                if dep in import_graph and module in import_graph[dep]:
                    circular_deps.append((module, dep))
        
        passed = len(circular_deps) == 0
        
        self.total_tests += 1
        if passed:
            self.passed_tests += 1
        else:
            self.failed_tests += 1
            
        return {
            'passed': passed,
            'circular_dependencies': circular_deps,
            'message': f"Circular dependencies: {len(circular_deps)} found"
        }
    
    def generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive final report."""
        success_rate = self.passed_tests / max(1, self.total_tests)
        
        # Determine overall quality gate status
        quality_gate_passed = (
            success_rate >= 0.8 and  # 80% of tests must pass
            self.failed_tests <= 5    # Maximum 5 critical failures
        )
        
        report = {
            'timestamp': time.time(),
            'overall_status': 'PASSED' if quality_gate_passed else 'FAILED',
            'success_rate': success_rate,
            'total_tests': self.total_tests,
            'passed_tests': self.passed_tests,
            'failed_tests': self.failed_tests,
            'detailed_results': self.results,
            'recommendations': self.generate_recommendations()
        }
        
        # Print summary
        print(f"📊 FINAL QUALITY GATE REPORT")
        print(f"Status: {'✅ PASSED' if quality_gate_passed else '❌ FAILED'}")
        print(f"Success Rate: {success_rate:.1%} ({self.passed_tests}/{self.total_tests} tests passed)")
        
        if not quality_gate_passed:
            print(f"\n🚨 Quality Gate Failed - {self.failed_tests} critical issues found")
        else:
            print(f"\n🎉 Quality Gate Passed - Ready for production deployment")
        
        return report
    
    def generate_recommendations(self) -> List[str]:
        """Generate improvement recommendations based on results."""
        recommendations = []
        
        # Code quality recommendations
        if self.results['code_quality']['syntax_check']['syntax_errors']:
            recommendations.append("Fix syntax errors before deployment")
        
        # Security recommendations
        security_results = self.results['security_scan']
        if security_results['hardcoded_secrets']['potential_secrets']:
            recommendations.append("Remove hardcoded secrets and use environment variables")
        
        if security_results['unsafe_functions']['unsafe_usage']:
            recommendations.append("Replace unsafe functions (eval, exec) with safer alternatives")
        
        # Performance recommendations
        perf_results = self.results['performance_check']
        if not perf_results['memory_usage']['passed']:
            recommendations.append("Optimize memory usage during module imports")
        
        # Documentation recommendations
        doc_results = self.results['documentation_check']
        if doc_results['docstring_coverage']['overall_coverage'] < 0.8:
            recommendations.append("Improve docstring coverage for better maintainability")
        
        # Default recommendations
        if not recommendations:
            recommendations = [
                "Consider adding more comprehensive tests",
                "Monitor performance in production",
                "Keep dependencies updated",
                "Regular security audits recommended"
            ]
        
        return recommendations


def main():
    """Main quality verification entry point."""
    project_root = os.getcwd()
    validator = QualityGateValidator(project_root)
    
    try:
        report = validator.run_all_quality_gates()
        
        # Save report
        report_file = Path(project_root) / 'quality_gate_report.json'
        with open(report_file, 'w') as f:
            import json
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n📋 Detailed report saved to: {report_file}")
        
        # Exit with appropriate code
        exit_code = 0 if report['overall_status'] == 'PASSED' else 1
        return exit_code
        
    except Exception as e:
        print(f"\n💥 Quality verification failed with error: {e}")
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
