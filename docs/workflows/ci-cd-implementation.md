# CI/CD Implementation Guide

This document provides templates and guidance for implementing GitHub Actions workflows for Fugatto Audio Lab.

## Required Workflows

### 1. Main CI/CD Pipeline

Create `.github/workflows/ci.yml`:

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
  workflow_dispatch:

env:
  PYTHON_VERSION: "3.10"
  CUDA_VERSION: "12.2"

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
        cache: 'pip'
    
    - name: Install dependencies
      run: |
        pip install -e ".[dev]"
        
    - name: Run pre-commit
      uses: pre-commit/action@v3.0.0
      
    - name: Run tests
      run: |
        pytest --cov=fugatto_lab --cov-report=xml --cov-report=html
        
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  security:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Run security scan
      uses: pypa/gh-action-pip-audit@v1.0.8
      with:
        inputs: requirements.txt requirements-dev.txt
        
    - name: Run Bandit security linter
      run: |
        pip install bandit[toml]
        bandit -r fugatto_lab/ -f json -o bandit-report.json
        
    - name: Upload security results
      uses: github/codeql-action/upload-sarif@v2
      if: always()
      with:
        sarif_file: bandit-report.json

  build-docker:
    runs-on: ubuntu-latest
    needs: [test, security]
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
      
    - name: Build and test Docker image
      run: |
        docker build -t fugatto-audio-lab:latest .
        docker run --rm fugatto-audio-lab:latest python -c "import fugatto_lab; print('Package imports successfully')"
```

### 2. Release Automation

Create `.github/workflows/release.yml`:

```yaml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  release:
    runs-on: ubuntu-latest
    permissions:
      contents: write
      id-token: write
    
    steps:
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0
        
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: "3.10"
        
    - name: Install build dependencies
      run: |
        pip install build twine
        
    - name: Build package
      run: python -m build
      
    - name: Verify package
      run: twine check dist/*
      
    - name: Create GitHub Release
      uses: softprops/action-gh-release@v1
      with:
        files: dist/*
        generate_release_notes: true
        
    - name: Publish to PyPI
      uses: pypa/gh-action-pypi-publish@release/v1
```

### 3. Dependency Updates

Create `.github/workflows/dependency-update.yml`:

```yaml
name: Dependency Updates

on:
  schedule:
    - cron: '0 2 * * 1'  # Weekly on Monday
  workflow_dispatch:

jobs:
  update-dependencies:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
      with:
        token: ${{ secrets.GITHUB_TOKEN }}
        
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: "3.10"
        
    - name: Update dependencies
      run: |
        pip install pip-tools
        pip-compile --upgrade pyproject.toml
        
    - name: Create Pull Request
      uses: peter-evans/create-pull-request@v5
      with:
        title: "chore: update dependencies"
        body: "Automated dependency updates"
        branch: "chore/dependency-updates"
        commit-message: "chore: update dependencies"
```

## Implementation Steps

1. **Create workflows directory**:
   ```bash
   mkdir -p .github/workflows
   ```

2. **Add workflow files** using the templates above

3. **Configure secrets** in GitHub repository settings:
   - `CODECOV_TOKEN` for code coverage
   - `PYPI_API_TOKEN` for package publishing

4. **Set up branch protection rules**:
   - Require status checks to pass
   - Require branches to be up to date
   - Require review from code owners

5. **Configure CODEOWNERS file**:
   ```
   # Global owners
   * @danieleschmidt
   
   # Python code
   *.py @danieleschmidt
   
   # Configuration
   pyproject.toml @danieleschmidt
   ```

## Security Configuration

The workflows include:
- Dependency vulnerability scanning with pip-audit
- Security linting with Bandit
- SARIF report upload for GitHub Security tab
- Signed releases with GitHub's attestation

## Monitoring Integration

Add monitoring steps to workflows:
- Performance regression testing
- Docker image size tracking
- Build time optimization alerts

## Rollback Procedures

If workflows fail:
1. Check workflow logs in Actions tab
2. Temporarily disable failing checks
3. Fix issues in feature branch
4. Re-enable checks after validation

## Next Steps

1. Implement the main CI workflow first
2. Test with a small PR
3. Add release automation
4. Set up dependency updates
5. Configure monitoring alerts