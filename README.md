# Surgery

This project explores the effects of dimensionality reduction on TinyLlama by replacing the Wk, Wq, and Wv attention matrices with reconstructed versions from their PCA components.

## Experiment

In this branch, the Wk, Wq, and Wv matrices of TinyLlama are replaced with matrices reconstructed from their PCA components that explain different amounts of the original variance:
- 99% of original variance
- 97.5% of original variance 
- 95% of original variance

As the variance threshold decreases, the model's performance progressively degrades, demonstrating the trade-offs between model compression and capabilities.

## Setup

Make sure you have Python and [uv](https://github.com/astral-sh/uv) installed.

## Usage

### Run with default settings

```bash
make run
```

This runs the script with the default variance threshold (90%).

### Run with custom variance threshold

```bash
make run VARIANCE_THRESHOLD=0.95
```

You can specify any variance threshold between 0 and 1.

### Development mode

To automatically rerun the script when Python files change:

```bash
make watch
```

You can also specify a custom variance threshold in watch mode:

```bash
make watch VARIANCE_THRESHOLD=0.99
```

### Installation only

To just create the virtual environment and install dependencies:

```bash
make install
```
