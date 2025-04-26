# Surgery

This project explores the effects of dimensionality reduction on TinyLlama by replacing the Wk, Wq, and Wv attention matrices with reconstructed versions from their PCA components.

## Experiment

In this branch, the Wk, Wq, and Wv matrices of TinyLlama are replaced with matrices reconstructed from their PCA components that explain different amounts of the original variance:
- 99% of original variance ![image](https://github.com/user-attachments/assets/2b0b07a8-a949-46e4-ae37-620cbffce119)
- 97.5% of original variance ![image](https://github.com/user-attachments/assets/000ec881-07ca-49b5-9cef-661abbb132e5)
- 95% of original variance ![image](https://github.com/user-attachments/assets/d5fea694-6efb-4e4d-9eef-855a96e56052)

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
