import mlx.core as mx
import mlx.optimizers as optim
import mlx.utils as utils
from mlx_mamba_native.model import MambaLMHeadModel, MambaConfig
from mlx_mamba_native.train import convert_to_lora, make_train_step

def main():
    mx.random.seed(42)
    print("Fine-Tuning Demo: Native Mamba-3 with LoRA in MLX")
    print("---------------------------------------------")

    # Step 1: Initialize model
    config = MambaConfig(
        d_model=64,
        n_layer=2,
        vocab_size=128,
        ssm_cfg={"is_mimo": True, "mimo_rank": 2}
    )
    model = MambaLMHeadModel(config)

    # Step 2: Inject LoRA and freeze base parameters
    print("Converting model to LoRA (freezing base parameters)...")
    convert_to_lora(model, r=4, alpha=8.0)

    # Cast model to bfloat16 for mixed precision SFT
    print("Casting model parameters to bfloat16...")
    flat_params = utils.tree_flatten(model.parameters())
    cast_params = [(k, v.astype(mx.bfloat16)) for k, v in flat_params]
    model.update(utils.tree_unflatten(cast_params))

    # Inspect parameter counts to verify freezing
    all_params = dict(utils.tree_flatten(model.parameters()))
    trainable_params = dict(utils.tree_flatten(model.trainable_parameters()))
    print(f"Total parameters (arrays): {len(all_params)}")
    print(f"Trainable parameters (arrays): {len(trainable_params)} (LoRA keys only)")
    print("Trainable parameters list:")
    for k in trainable_params.keys():
        print(f"  - {k}")

    # Step 3: Create synthetic toy language modeling data
    # Batch size = 4, Sequence length = 16
    B, L = 4, 16
    vocab_size = config.vocab_size
    
    # Inputs: random integer tokens
    inputs = mx.random.randint(low=1, high=vocab_size-1, shape=(B, L))
    # Targets: inputs shifted by 1
    targets = mx.concatenate([inputs[:, 1:], mx.random.randint(low=1, high=vocab_size-1, shape=(B, 1))], axis=1)

    # Step 4: Set up optimizer and compiled training step
    optimizer = optim.Adam(learning_rate=1e-3)
    train_step = make_train_step(model, optimizer)

    # Step 5: Run SFT loop for 50 steps
    print("\nRunning training loop for 50 steps...")
    initial_loss = None
    final_loss = None
    
    for step in range(1, 51):
        loss = train_step(inputs, targets)
        mx.eval(model.parameters(), optimizer.state)
        loss_val = loss.item()

        if step == 1:
            initial_loss = loss_val
            print(f"  Step {step:02d} | Loss: {loss_val:.4f}")
        elif step % 10 == 0:
            print(f"  Step {step:02d} | Loss: {loss_val:.4f}")
        
        final_loss = loss_val

    print("\nTraining summary:")
    print(f"  Initial Loss: {initial_loss:.4f}")
    print(f"  Final Loss:   {final_loss:.4f}")
    print(f"  Loss drop:    {initial_loss - final_loss:.4f}")

    assert final_loss < initial_loss, "Training failed: Loss did not decrease"
    print("\nFine-tuning completed successfully! LoRA parameter optimization validated.")


if __name__ == "__main__":
    main()
