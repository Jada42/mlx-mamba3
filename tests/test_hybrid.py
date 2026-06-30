import unittest
import numpy as np
import mlx.core as mx
from mlx_mamba_native.model import MambaLMHeadModel, MambaConfig
from mlx_mamba_native.cache import MambaCache
from mlx_mamba_native.generate import generate

class TestHybridModel(unittest.TestCase):

    def setUp(self):
        mx.random.seed(42)

    def test_hybrid_forward_and_step(self):
        # 1. Instantiate a hybrid model
        # 4 layers: layer 0 & 2 are Mamba-3, layer 1 & 3 are Attention
        config = MambaConfig(
            d_model=128,
            n_layer=4,
            vocab_size=1000,
            ssm_cfg={"is_mimo": False},
            attn_layer_idx=[1, 3],
            attn_cfg={"num_heads": 4}
        )
        model = MambaLMHeadModel(config)

        # Verify block types
        self.assertEqual(model.layers[0].__class__.__name__, "MambaBlock")
        self.assertEqual(model.layers[1].__class__.__name__, "AttentionBlock")
        self.assertEqual(model.layers[2].__class__.__name__, "MambaBlock")
        self.assertEqual(model.layers[3].__class__.__name__, "AttentionBlock")

        # 2. Run forward pass
        B, L = 2, 8
        prompt = mx.random.randint(1, 999, (B, L))
        logits_full = model(prompt)
        self.assertEqual(logits_full.shape, (B, L, model.vocab_size))
        print("Hybrid forward output shape:", logits_full.shape)

        # 3. Check step vs prefill equivalence
        # Sub-sequence: prompt (L=4) and next token
        prompt_sub = prompt[:, :4]
        next_token = prompt[:, 4:5]
        full_sub = prompt[:, :5]

        # Method A: Full prefill
        logits_a = model(full_sub)

        # Method B: Prefill then step
        cache = MambaCache.from_model(model, batch_size=B)
        _ = model(prompt_sub, cache=cache)  # Prefill
        
        # Step once
        logits_b = model.step(next_token, cache=cache)

        # Compare outputs at position t=4
        np_a = np.array(logits_a[:, 4, :])
        np_b = np.array(logits_b)

        diff = np.abs(np_a - np_b).max()
        print(f"Hybrid prefill-step max diff: {diff}")
        self.assertTrue(diff < 1e-5, f"Hybrid step equivalence check failed: {diff}")

    def test_hybrid_generation(self):
        # Verify that autoregressive generation helper works with hybrid models
        config = MambaConfig(
            d_model=64,
            n_layer=2,
            vocab_size=128,
            attn_layer_idx=[1],
            attn_cfg={"num_heads": 2}
        )
        model = MambaLMHeadModel(config)

        prompt = [5, 10, 15]
        output = generate(model, prompt, temp=0.0, max_tokens=5)
        self.assertEqual(output.shape, (1, 5))
        print("Hybrid generated sequence:", output.tolist()[0])


if __name__ == "__main__":
    unittest.main()
