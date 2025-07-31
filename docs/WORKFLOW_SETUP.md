# GitHub Workflows Setup Guide

## Overview

This repository includes production-ready GitHub Actions workflow templates that need to be manually activated by a repository administrator due to security policies.

## Quick Setup (5 minutes)

### 1. Copy Workflow Files

```bash
# From repository root
cp docs/workflows/production-ci-cd.yml .github/workflows/ci.yml
cp docs/workflows/production-release.yml .github/workflows/release.yml
```

### 2. Enable Repository Security Features

Go to **Settings → Security & Analysis** and enable:
- ✅ **Dependency graph**
- ✅ **Dependabot alerts**  
- ✅ **Dependabot security updates**
- ✅ **CodeQL analysis**
- ✅ **Secret scanning**
- ✅ **Push protection**

### 3. Configure Repository Secrets

Go to **Settings → Secrets and Variables → Actions** and add:

**Required Secrets:**
- `CODECOV_TOKEN` - Get from [codecov.io](https://codecov.io/) after connecting repository
- `PYPI_API_TOKEN` - Generate from [PyPI Account Settings](https://pypi.org/manage/account/token/)

**Optional Secrets (for advanced features):**
- `SLACK_WEBHOOK_URL` - For build notifications
- `DISCORD_WEBHOOK_URL` - For release announcements

### 4. Set Up Branch Protection

Go to **Settings → Branches** and add protection rule for `main`:
- ✅ **Require a pull request before merging**
- ✅ **Require status checks to pass before merging**
  - Select: `test`, `lint`, `security`
- ✅ **Require branches to be up to date before merging**
- ✅ **Restrict pushes that create files**

## Workflow Features

### CI Pipeline (`ci.yml`)
- **Multi-Python Testing**: Tests against Python 3.10, 3.11, 3.12
- **Multi-PyTorch Testing**: Tests against PyTorch 2.3.0, 2.4.0
- **Security Scanning**: Bandit, Safety, CodeQL, Trivy
- **Code Quality**: Black, isort, ruff, mypy
- **Coverage Reporting**: Automatic Codecov integration
- **Performance Benchmarking**: Tracks performance regressions
- **Docker Validation**: Builds and tests container images

### Release Pipeline (`release.yml`)
- **Automated Testing**: Full test suite before release
- **Package Building**: Builds Python wheel and source distributions
- **Docker Publishing**: Pushes to GitHub Container Registry
- **PyPI Publishing**: Automatic package publishing
- **GitHub Releases**: Auto-generated release notes and changelog

## Advanced Configuration

### Custom Environment Variables

Add to workflow files as needed:
```yaml
env:
  FUGATTO_MODEL_CACHE_DIR: ./cache/models
  FUGATTO_MAX_BATCH_SIZE: 4
  TORCH_CUDA_ARCH_LIST: "7.0 7.5 8.0 8.6"
```

### Matrix Testing Customization

Modify the test matrix in `ci.yml`:
```yaml
strategy:
  matrix:
    python-version: ['3.10', '3.11', '3.12']
    torch-version: ['2.3.0', '2.4.0']
    os: [ubuntu-latest, windows-latest, macos-latest]  # Add OS matrix
```

### GPU Testing

Add GPU runners for CUDA testing:
```yaml
test-gpu:
  runs-on: self-hosted-gpu  # Requires self-hosted GPU runner
  steps:
    - name: Test CUDA functionality
      run: python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## Monitoring and Alerts

### CodeCov Integration
1. Visit [codecov.io](https://codecov.io/)
2. Connect your GitHub repository
3. Copy the repository token to `CODECOV_TOKEN` secret

### Security Alerts
GitHub will automatically:
- ✅ Scan for vulnerable dependencies
- ✅ Create security advisories for detected issues
- ✅ Generate Dependabot PRs for updates
- ✅ Block commits containing secrets

## Troubleshooting

### Common Issues

**Workflow Permission Errors:**
```
Error: Resource not accessible by integration
```
**Solution**: Ensure workflows have necessary permissions in the YAML files.

**Test Failures:**
```
Error: Tests failed with exit code 1
```
**Solution**: Run tests locally first: `make test`

**Docker Build Failures:**
```
Error: Docker build failed
```
**Solution**: Test Docker build locally: `make docker-build`

### Debug Mode

Enable debug logging by setting repository variable:
- Go to **Settings → Variables → Actions**
- Add variable: `ACTIONS_STEP_DEBUG` = `true`

## Security Best Practices

### Secrets Management
- ✅ Never commit secrets to repository
- ✅ Use repository secrets for sensitive data
- ✅ Rotate tokens regularly
- ✅ Use principle of least privilege

### Workflow Security
- ✅ Pin action versions to specific commits
- ✅ Review third-party actions before use
- ✅ Use `pull_request_target` carefully
- ✅ Validate inputs and outputs

## Performance Optimization

### Cache Strategy
The workflows include automatic caching for:
- Python dependencies (`pip cache`)
- Docker layers (`buildx cache`)
- PyTorch models (custom cache)

### Parallel Execution
- Tests run in parallel across Python versions
- Security scans run concurrently with tests
- Docker builds use BuildKit for optimization

## Support

- **Documentation**: See `docs/` directory
- **Issues**: Create GitHub issue with `ci/cd` label
- **Security**: Follow `SECURITY.md` for vulnerability reports

## Manual Workflow Triggers

You can manually trigger workflows:

```bash
# Trigger CI pipeline
gh workflow run ci.yml

# Trigger with specific inputs
gh workflow run ci.yml -f python-version=3.11

# Trigger release (requires tag)
git tag v1.0.0
git push origin v1.0.0
```

This setup provides enterprise-grade CI/CD capabilities with comprehensive testing, security scanning, and automated releases while maintaining security and compliance requirements.