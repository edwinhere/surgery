from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

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

def main():
    print("Analyzing and evaluating TinyLlama model...")
    model, tokenizer = download_and_analyze_tinyllama()
    
    # Evaluate model
    score = evaluate_tinyllama(model, tokenizer)
    
    print("\nEvaluation complete!")
    print(f"Functionality Score: {score:.1f}%")

if __name__ == "__main__":
    main()
