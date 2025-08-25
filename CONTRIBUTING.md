# Contributing to Tennis-Vision

Thank you for your interest in contributing to Tennis-Vision! This document provides guidelines and information for contributors.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Basic understanding of computer vision and machine learning
- Familiarity with PyTorch and OpenCV

### Development Setup

1. **Fork the repository**
   ```bash
   # Fork on GitHub, then clone your fork
   git clone https://github.com/HarshTomar1234/Tennis-Vision.git
   cd Tennis-Vision
   ```

2. **Set up development environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Install development dependencies
   pip install pytest black flake8 pre-commit
   ```

3. **Set up pre-commit hooks**
   ```bash
   pre-commit install
   ```

## How to Contribute

### Reporting Issues

Before creating an issue, please:
- Check existing issues to avoid duplicates
- Use the issue templates when available
- Provide clear, detailed descriptions
- Include system information and error logs

### Submitting Pull Requests

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Follow the code style guidelines below
   - Add tests for new functionality
   - Update documentation as needed

3. **Test your changes**
   ```bash
   # Run tests
   pytest tests/
   
   # Check code style
   black --check .
   flake8 .
   ```

4. **Commit and push**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   git push origin feature/your-feature-name
   ```

5. **Create Pull Request**
   - Use descriptive titles and descriptions
   - Reference related issues
   - Include screenshots/videos for visual changes

## Code Style Guidelines

### Python Code Style

- Follow PEP 8 standards
- Use Black for code formatting
- Maximum line length: 88 characters
- Use type hints where appropriate

```python
# Good
def detect_players(frame: np.ndarray, confidence: float = 0.7) -> List[Dict]:
    """Detect players in the given frame."""
    pass

# Bad
def detect_players(frame, confidence=0.7):
    pass
```

### Documentation

- Use docstrings for all functions and classes
- Follow Google-style docstrings
- Include parameter types and return types
- Add usage examples for complex functions

```python
def track_ball(frames: List[np.ndarray], model_path: str) -> List[Tuple[int, int]]:
    """Track ball positions across multiple frames.
    
    Args:
        frames: List of video frames as numpy arrays
        model_path: Path to the trained ball detection model
        
    Returns:
        List of (x, y) coordinates for ball positions
        
    Example:
        >>> frames = load_video_frames("match.mp4")
        >>> positions = track_ball(frames, "models/ball_model.pt")
        >>> print(f"Ball detected at {len(positions)} positions")
    """
    pass
```

### Commit Message Format

Use conventional commits format:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

Examples:
```
feat(tracker): add ball trajectory smoothing
fix(detection): resolve false positive filtering
docs(readme): update installation instructions
```

## Development Areas

### Priority Areas for Contribution

1. **Model Improvements**
   - Better ball detection in challenging conditions
   - Enhanced player tracking accuracy
   - Shot classification refinements

2. **Performance Optimization**
   - GPU acceleration improvements
   - Memory usage optimization
   - Real-time processing enhancements

3. **New Features**
   - Additional shot types detection
   - Player statistics analysis
   - Match highlights generation

4. **Testing & Quality**
   - Unit tests for core modules
   - Integration tests
   - Performance benchmarks

### Model Training Guidelines

If contributing model improvements:

1. **Dataset Requirements**
   - Minimum 500 annotated samples
   - Diverse lighting and court conditions
   - Proper train/validation/test splits

2. **Training Standards**
   - Document hyperparameters
   - Include training logs and metrics
   - Provide model evaluation results

3. **Model Submission**
   - Include model weights and configuration
   - Provide comparison with existing models
   - Document inference requirements

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_ball_tracker.py

# Run with coverage
pytest --cov=./ --cov-report=html
```

### Writing Tests

- Write tests for all new functions
- Use descriptive test names
- Include edge cases and error conditions
- Mock external dependencies

```python
def test_ball_detection_with_valid_frame():
    """Test ball detection with a valid input frame."""
    frame = create_test_frame_with_ball()
    detections = detect_ball(frame)
    assert len(detections) == 1
    assert detections[0]['confidence'] > 0.5
```

## Documentation

### Updating Documentation

- Update README.md for new features
- Add docstrings to new functions
- Update API documentation
- Include usage examples

### Documentation Build

```bash
# Generate documentation (if using Sphinx)
cd docs/
make html
```

## Community Guidelines

### Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Help newcomers learn
- Focus on technical merit

### Getting Help

- Check existing documentation first
- Search closed issues for solutions
- Ask questions in discussions
- Tag maintainers for urgent issues

## Release Process

### Version Numbering

We follow Semantic Versioning (SemVer):
- MAJOR: Breaking changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes

### Release Checklist

- [ ] Update version numbers
- [ ] Update CHANGELOG.md
- [ ] Run full test suite
- [ ] Update documentation
- [ ] Create release notes

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Credited in academic papers (if applicable)

## Questions?

Feel free to:
- Open a discussion for general questions
- Create an issue for bugs or feature requests
- Contact maintainers directly for sensitive matters

Thank you for contributing to Tennis-Vision! 🎾
