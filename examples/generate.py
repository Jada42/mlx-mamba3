import os
import mlx.core as mx
from mlx_mamba_native.model import MambaLMHeadModel, MambaConfig
from mlx_mamba_native.generate import generate
from mlx_mamba_native.weights import save_weights, load_weights

def main():
    mx.random.seed(42)
    print("Initializing Mamba-3 model (MIMO mode)...")

    # Small configuration suitable for demonstration
    config = MambaConfig(
        d_model=128,
        n_layer=4,
        vocab_size=1000,
        ssm_cfg={"is_mimo": True, "mimo_rank": 2}
    )
    model = MambaLMHeadModel(config)

    # 1. Autoregressive text generation demo
    prompt = [10, 20, 30, 40]
    print(f"Prompt token IDs: {prompt}")
    print("Generating 10 completion tokens...")
    output = generate(model, prompt, temp=0.0, max_tokens=10)
    print(f"Generated sequence of token IDs: {output.tolist()[0]}")

    # 2. Weights saving and loading demo
    weight_path = "demo_weights.safetensors"
    print(f"Saving weights to '{weight_path}'...")
    save_weights(model, weight_path)

    print("Re-initializing a fresh model and loading weights back...")
    new_model = MambaLMHeadModel(config)
    load_weights(new_model, weight_path)

    # Verify that the loaded model outputs the exact same logits
    x = mx.array([[10, 20, 30, 40]])
    logits_old = model(x)
    logits_new = new_model(x)
    diff = mx.max(mx.abs(logits_old - logits_new)).item()
    print(f"Max absolute difference between original and loaded model: {diff}")
    assert diff == 0, "Weights load verification failed"

    # Cleanup weights file
    if os.path.exists(weight_path):
        os.remove(weight_path)
        print(f"Cleaned up '{weight_path}'")

    print("\nGeneration and serialization example completed successfully!")


if __name__ == "__main__":
    main()
