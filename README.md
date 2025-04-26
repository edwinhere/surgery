# Surgery

This project explores the effects of dimensionality reduction on TinyLlama by replacing the Wk, Wq, and Wv attention matrices with reconstructed versions from their PCA components.

## Trivial Result

In the `pca` branch, the Wk, Wq, and Wv matrices of TinyLlama are replaced with matrices reconstructed from their PCA components that explain different amounts of the original variance:
- 99% of original variance ![image](https://github.com/user-attachments/assets/2b0b07a8-a949-46e4-ae37-620cbffce119)
- 97.5% of original variance ![image](https://github.com/user-attachments/assets/000ec881-07ca-49b5-9cef-661abbb132e5)
- 95% of original variance ![image](https://github.com/user-attachments/assets/d5fea694-6efb-4e4d-9eef-855a96e56052)

As the variance threshold decreases, the model's performance progressively degrades, demonstrating the trade-offs between model compression and capabilities.

## Interesting Result

In the branch `shared_kv_pca` For each layer:

1. The code now groups parameters by layer name (e.g., `model.layers.0.self_attn`)
2. For each layer:
   - `q_proj` is processed separately with its own PCA
   - `k_proj` and `v_proj` are processed together with a shared PCA:
     - The matrices are stacked vertically using `np.vstack`
     - A single PCA is fit on the combined matrix
     - The same PCA components are then used to transform and reconstruct each matrix separately

This approach ensures that `k_proj` and `v_proj` matrices use the same principal components, potentially capturing related patterns across both matrices while still maintaining their individual characteristics.

- 99% of original variance
- 90% of original variance
- 80% of original variance
- 70% of original variance
- 60% of original variance
- 50% of original variance
- 40% of original variance
- 30% of original variance
- 20% of original variance

Notice how when we `np.vstack`, the `k_proj` and `v_proj` matrices and calculate a shared PCA used to reconstruct `k_proj` and `v_proj` the LLM is much more resilient to PCA variance threshold.

## Conclusion & Further Direction

Perhaps this is how teams competing to make foundation models derive their insight that latent matrices which help create `W_k` and `W_v` share structure. Also it seems like KV has a lot more redundancy than I thought. Perhaps the next step is to reconstruct from PCA components shared between layers.

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
