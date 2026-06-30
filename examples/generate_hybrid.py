import mlx.core as mx
from mlx_mamba_native.model import MambaLMHeadModel, MambaConfig
from mlx_mamba_native.generate import generate

def main():
    mx.random.seed(42)
    print("Initializing a hybrid Transformer-Mamba-3 model...")

    # 4-layer configuration:
    # - Layer 0 & 2: Mamba-3 SISO blocks
    # - Layer 1 & 3: Causal Attention blocks (Self-Attention)
    config = MambaConfig(
        d_model=128,
        n_layer=4,
        vocab_size=1000,
        attn_layer_idx=[1, 3],
        attn_cfg={"num_heads": 4}
    )
    model = MambaLMHeadModel(config)

    # Autoregressive generation demo
    prompt = [10, 20, 30, 40]
    print(f"Prompt token IDs: {prompt}")
    print("Generating 10 completion tokens using the hybrid architecture...")
    output = generate(model, prompt, temp=0.0, max_tokens=10)
    print(f"Generated sequence of token IDs: {output.tolist()[0]}")
    print("\nHybrid model generation completed successfully!")


if __name__ == "__main__":
    main()
