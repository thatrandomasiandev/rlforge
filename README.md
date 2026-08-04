# RLForge

A production-ready, research-friendly reinforcement learning library built on NumPy, Gymnasium, and (optionally) PyTorch.

## Status

Early alpha (`0.1.0`). Core utilities and environment wrappers are in place; algorithm APIs are evolving.

## Requirements

- Python 3.10+
- NumPy, Gymnasium
- Optional: PyTorch 2.0+ for deep RL algorithms

## Install

```bash
# editable install (recommended while developing)
pip install -e .

# with PyTorch + test tooling
pip install -e ".[dev]"
```

## Quick start

```python
import rlforge

print(rlforge.__version__)
```

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT — see [LICENSE](LICENSE).
