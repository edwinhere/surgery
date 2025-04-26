from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
from sklearn.decomposition import PCA
import copy
import argparse

def download_and_analyze_tinyllama():
    # Download and load the TinyLlama model
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    print(f"Downloading {model_name}...")
    
    # Load model with lower precision to save memory
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # List all model layers
    print("\nTinyLlama Model Layers:")
    print("=" * 40)
    
    # Get the named modules
    for name, module in model.named_modules():
        # Skip the parent module
        if name == "":
            continue
        
        # Format module info
        module_type = module.__class__.__name__
        param_count = sum(p.numel() for p in module.parameters() if p.requires_grad)
        
        if param_count > 0:  # Only show layers with parameters
            print(f"{name:<30} | {module_type:<20} | {param_count:,} params")
    
    # Print total parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\nTotal parameters:", f"{total_params:,}")
    
    return model, tokenizer

def evaluate_tinyllama(model, tokenizer):
    """
    Simple evaluation to check if TinyLlama is functioning properly.
    """
    print("\nEvaluating TinyLlama with a simple text generation task...")
    
    # Simple prompt for evaluation
    prompt = "<|system|>\nYou are a helpful assistant.\n<|user|>\nWhat is machine learning?\n<|assistant|>"
    
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate response
    print("Generating response...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
    
    # Decode and print the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    print("\nPrompt + Response:")
    print("-" * 40)
    print(response)
    print("-" * 40)
    
    # Simple score based on response length (just as a basic check)
    words = response.split()
    score = min(len(words) / 50, 1.0) * 100  # Normalize to 0-100%
    
    print(f"\nSimple Functionality Score: {score:.1f}%")
    print("(This score simply checks if the model can generate a reasonable length response)")
    
    return score

def apply_pca_to_matrix(weight_matrix, variance_threshold=0.99):
    """
    Apply PCA to a weight matrix, keeping enough components to explain 
    the specified amount of variance (default: 99%)
    """
    original_shape = weight_matrix.shape
    original_device = weight_matrix.device
    original_dtype = weight_matrix.dtype
    
    # Move to CPU and convert to float32 for PCA
    weight_np = weight_matrix.detach().cpu().float().numpy()
    
    # For 2D matrices (e.g., linear layers)
    if len(original_shape) == 2:
        # Apply PCA to the larger dimension
        if original_shape[0] <= original_shape[1]:
            # More columns than rows, apply PCA to columns
            pca = PCA(n_components=min(original_shape[1], original_shape[0]), svd_solver='full')
            pca.fit(weight_np)
            # Find number of components for desired variance
            n_components = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= variance_threshold) + 1
            print(f"  Using {n_components} components to explain {variance_threshold*100:.1f}% variance")
            # Transform and reconstruct with selected components
            transformed = pca.transform(weight_np)[:, :n_components]
            reconstructed = np.matmul(transformed, pca.components_[:n_components])
        else:
            # More rows than columns, apply PCA to rows
            pca = PCA(n_components=min(original_shape[0], original_shape[1]), svd_solver='full')
            pca.fit(weight_np.T)
            # Find number of components for desired variance
            n_components = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= variance_threshold) + 1
            print(f"  Using {n_components} components to explain {variance_threshold*100:.1f}% variance")
            # Transform and reconstruct with selected components
            transformed = pca.transform(weight_np.T)[:, :n_components]
            reconstructed = np.matmul(transformed, pca.components_[:n_components]).T
    
    # For embedding matrices or attention matrices (common in transformers)
    elif len(original_shape) == 3:
        # Reshape to 2D, apply PCA, then reshape back
        reshaped = weight_np.reshape(original_shape[0], -1)
        max_components = min(reshaped.shape[0], reshaped.shape[1])
        pca = PCA(n_components=max_components, svd_solver='full')
        pca.fit(reshaped)
        # Find number of components for desired variance
        n_components = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= variance_threshold) + 1
        print(f"  Using {n_components} components to explain {variance_threshold*100:.1f}% variance")
        # Transform and reconstruct with selected components
        transformed = pca.transform(reshaped)[:, :n_components]
        reconstructed_2d = np.matmul(transformed, pca.components_[:n_components])
        reconstructed = reconstructed_2d.reshape(original_shape)
    
    # Return as tensor on original device with original dtype
    reconstructed_tensor = torch.tensor(reconstructed, device=original_device, dtype=original_dtype)
    return reconstructed_tensor

def compress_model_with_pca(model, variance_threshold=0.99):
    """
    Apply PCA to q_proj, k_proj, v_proj matrices in attention layers,
    keeping enough components to explain the specified variance threshold
    """
    # Create a deep copy to avoid modifying the original model
    compressed_model = copy.deepcopy(model)
    
    print(f"\nApplying PCA with variance threshold {variance_threshold*100:.1f}% to attention projection matrices...")
    
    # Track compression stats
    total_processed = 0
    
    # Process only q_proj, k_proj, v_proj matrices
    for name, param in compressed_model.named_parameters():
        # Only process attention projection matrices
        if any(proj in name for proj in ['q_proj', 'k_proj', 'v_proj']):
            print(f"Applying PCA to {name} with shape {param.shape}...")
            
            # Apply PCA and update the parameter
            with torch.no_grad():
                param.copy_(apply_pca_to_matrix(param, variance_threshold))
            total_processed += 1
    
    print(f"PCA applied to {total_processed} attention projection matrices.")
    return compressed_model
            
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Analyze and evaluate TinyLlama model with PCA compression")
    parser.add_argument("--variance-threshold", type=float, default=0.90,
                        help="Variance threshold for PCA compression (default: 0.90)")
    args = parser.parse_args()

    print("Analyzing and evaluating TinyLlama model...")
    model, tokenizer = download_and_analyze_tinyllama()
    
    # Evaluate original model
    print("\n=== Original Model Evaluation ===")
    original_score = evaluate_tinyllama(model, tokenizer)
    
    # Compress model with PCA
    compressed_model = compress_model_with_pca(model, variance_threshold=args.variance_threshold)
    
    # Print compressed model layers
    print("\nCompressed TinyLlama Model Layers:")
    print("=" * 40)
    
    for name, module in compressed_model.named_modules():
        # Skip the parent module
        if name == "":
            continue
        
        # Format module info
        module_type = module.__class__.__name__
        param_count = sum(p.numel() for p in module.parameters() if p.requires_grad)
        
        if param_count > 0:  # Only show layers with parameters
            print(f"{name:<30} | {module_type:<20} | {param_count:,} params")
    
    # Print total parameters
    total_params = sum(p.numel() for p in compressed_model.parameters() if p.requires_grad)
    print("\nTotal parameters in compressed model:", f"{total_params:,}")
    
    # Evaluate compressed model
    print(f"\n=== PCA Compressed Model Evaluation ({args.variance_threshold*100}% variance) ===")
    compressed_score = evaluate_tinyllama(compressed_model, tokenizer)
    
    # Compare results
    print("\n=== Comparison ===")
    print(f"Original model score: {original_score:.1f}%")
    print(f"Compressed model score: {compressed_score:.1f}%")
    print(f"Score difference: {compressed_score - original_score:.1f}%")
    
    print("\nEvaluation complete!")

if __name__ == "__main__":
    main()
