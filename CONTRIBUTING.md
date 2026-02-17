# Contributing to REINVENT4 Meta-Controller

Thanks for considering a contribution. This project welcomes PRs for:

- New arm strategies
- Improved bandit policies
- Better default configurations
- Documentation improvements
- Bug fixes

## Development Setup

```bash
git clone https://github.com/yourusername/reinvent4-meta-controller.git
cd reinvent4-meta-controller
pip install -e ".[dev]"
```

## Before Submitting a PR

1. **Run tests:**
   ```bash
   make test
   ```

2. **Lint your code:**
   ```bash
   make lint
   ```

3. **Format code:**
   ```bash
   make format
   ```

4. **Run the demo:**
   ```bash
   make demo
   ```

5. **Update documentation** if you've changed APIs or added features

## Code Style

- Follow PEP 8 (enforced by ruff)
- Use type hints where practical
- Write docstrings for public functions
- Keep functions focused and small

## Writing Tests

- Add tests for new features in `tests/`
- Use pytest fixtures for common setup
- Mark slow tests with `@pytest.mark.slow`
- Aim for >80% coverage

Example:

```python
import pytest

def test_new_feature():
    """Test description."""
    result = my_function(input_data)
    assert result == expected_output
```

## Documentation

- Update `docs/` for substantial changes
- Keep README examples up to date
- Use British English spelling
- Be direct and practical (avoid AI-sounding prose)

## Commit Messages

Use clear, descriptive commit messages:

```
Add scaffold diversity metric to reporting

- Implements Bemis-Murcko scaffold extraction
- Adds diversity calculation to episode summaries
- Updates report template to show scaffold counts
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes
4. Run all checks (`make check`)
5. Commit your changes
6. Push to your fork
7. Open a PR with a clear description

### PR Description Template

```markdown
## What does this PR do?

Brief description of the changes.

## Why is this needed?

Context about the problem being solved.

## How was this tested?

- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Demo runs successfully

## Checklist

- [ ] Tests pass
- [ ] Linting passes
- [ ] Documentation updated
- [ ] CHANGELOG.md updated (if applicable)
```

## Code of Conduct

- Be respectful and professional
- Welcome newcomers
- Focus on constructive feedback
- Acknowledge contributions

## Questions?

Open an issue or discussion on GitHub.

---

**Note**: This is a research prototype → production pipeline. Breaking changes may occur. We'll do our best to communicate them clearly.
