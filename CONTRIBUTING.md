# Contributing to Vertebral Body Segmentation

Thank you for your interest in contributing to this project! This document provides guidelines for contributing to the Vertebral Body Segmentation repository.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Code Style](#code-style)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/Vertebral-Body-Segmentation.git
   cd Vertebral-Body-Segmentation
   ```
3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/farhatmasood/Vertebral-Body-Segmentation.git
   ```

## Development Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install development dependencies (optional):
   ```bash
   pip install black flake8 pytest
   ```

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:
- Clear description of the bug
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, PyTorch version)
- Error messages or logs

### Suggesting Enhancements

For feature requests, create an issue with:
- Clear description of the proposed feature
- Use case and motivation
- Proposed implementation approach (optional)

### Code Contributions

1. **Create a branch** for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**:
   - Follow the code style guidelines
   - Add tests if applicable
   - Update documentation as needed

3. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Description of changes"
   ```

4. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Create a Pull Request** from your fork to the main repository

## Code Style

### Python Style Guidelines

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and concise
- Maximum line length: 100 characters

### Docstring Format

Use Google-style docstrings:

```python
def function_name(param1, param2):
    """
    Brief description of function.
    
    Longer description if needed.
    
    Args:
        param1 (type): Description of param1
        param2 (type): Description of param2
    
    Returns:
        type: Description of return value
    
    Raises:
        ValueError: Description of when this error is raised
    """
    pass
```

### Code Formatting

We recommend using `black` for automatic code formatting:

```bash
black .
```

### Linting

Check your code with `flake8`:

```bash
flake8 --max-line-length=100 .
```

## Testing

### Running Tests

If tests are available, run them with:

```bash
pytest tests/
```

### Adding Tests

When adding new features, please include tests:

1. Create test file in `tests/` directory
2. Name test files as `test_*.py`
3. Name test functions as `test_*`
4. Use descriptive test names

Example:

```python
def test_dice_coefficient():
    """Test dice coefficient calculation"""
    pred = torch.ones(1, 256, 256)
    target = torch.ones(1, 256, 256)
    dice = dice_coefficient(pred, target)
    assert dice == 1.0
```

## Pull Request Process

1. **Update documentation**: Ensure README and docstrings are updated
2. **Add tests**: Include tests for new functionality
3. **Update CHANGELOG**: Add your changes to the changelog (if exists)
4. **Check CI**: Ensure all CI checks pass
5. **Request review**: Request review from maintainers
6. **Address feedback**: Make requested changes
7. **Merge**: Maintainers will merge once approved

### Pull Request Checklist

- [ ] Code follows the project's style guidelines
- [ ] Added tests for new functionality
- [ ] All tests pass
- [ ] Documentation updated
- [ ] Commit messages are clear and descriptive
- [ ] No merge conflicts with main branch

## Types of Contributions We're Looking For

- **Bug fixes**: Fix existing issues
- **New features**: Add new model architectures, metrics, or utilities
- **Documentation**: Improve README, add tutorials, or examples
- **Performance**: Optimize existing code
- **Tests**: Add or improve test coverage
- **Examples**: Add example scripts or notebooks

## Areas for Contribution

### Model Improvements
- Additional segmentation architectures (Attention UNet, UNet++, etc.)
- 3D segmentation support
- Multi-class segmentation
- Transfer learning approaches

### Data Processing
- Additional data augmentation techniques
- Support for more medical image formats
- Data validation utilities
- Pre-processing pipelines

### Training & Evaluation
- Additional loss functions
- More evaluation metrics
- Mixed precision training
- Distributed training support

### Visualization
- Interactive visualization tools
- 3D visualization support
- Tensorboard improvements
- Web-based result viewer

### Documentation
- Tutorial notebooks
- Video tutorials
- API documentation
- Usage examples

## Code Review Process

Maintainers will review your PR and may:
- Request changes
- Ask questions for clarification
- Suggest improvements
- Approve and merge

Please be patient and responsive during the review process.

## Community Guidelines

- Be respectful and constructive
- Help others when you can
- Follow the code of conduct
- Give credit where it's due

## Questions?

If you have questions:
1. Check existing issues and discussions
2. Create a new issue with the "question" label
3. Contact the maintainers

## License

By contributing, you agree that your contributions will be licensed under the GNU General Public License v3.0.

## Acknowledgments

Thank you for contributing to advancing medical image segmentation research!

---

For more information, see the [README.md](README.md) and the [paper](https://www.sciencedirect.com/science/article/abs/pii/S1746809421008272).
