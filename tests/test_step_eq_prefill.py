import unittest
import numpy as np
import mlx.core as mx
from mlx_mamba_native.model import MambaLMHeadModel, MambaConfig
from mlx_mamba_native.cache import MambaCache

class TestStepPrefillEquivalence(unittest.TestCase):

    def setUp(self):
        mx.random.seed(42)

    def test_siso_equivalence(self):
        config = MambaConfig(
            d_model=128,
            n_layer=2,
            vocab_size=1000,
            ssm_cfg={"is_mimo": False}
        )
        model = MambaLMHeadModel(config)

        # 1. Inputs
        B = 1
        prompt = mx.array([[1, 2, 3, 4]])
        next_token = mx.array([[5]])
        full_seq = mx.concatenate([prompt, next_token], axis=1)  # shape (1, 5)

        # 2. Method A: Full-sequence prefill
        logits_full = model(full_seq)  # shape (1, 5, vocab_size)

        # 3. Method B: Prefill then Step
        cache = MambaCache.from_model(model, batch_size=B)
        _ = model(prompt, cache=cache)  # Populates cache with t=0..3 states

        # Run step recurrence for t=4 (next_token)
        logits_step = model.step(next_token, cache=cache)  # shape (1, vocab_size)

        # 4. Compare outputs
        # The logits for the last token in full prefill (t=4) should match the step output
        np_full = np.array(logits_full[:, 4, :])
        np_step = np.array(logits_step)

        diff = np.abs(np_full - np_step).max()
        print(f"SISO prefill-step max diff: {diff}")
        self.assertTrue(diff < 1e-5, f"SISO prefill-step mismatch: {diff}")

    def test_mimo_equivalence(self):
        config = MambaConfig(
            d_model=128,
            n_layer=2,
            vocab_size=1000,
            ssm_cfg={"is_mimo": True, "mimo_rank": 2}
        )
        model = MambaLMHeadModel(config)

        # 1. Inputs
        B = 1
        prompt = mx.array([[1, 2, 3, 4]])
        next_token = mx.array([[5]])
        full_seq = mx.concatenate([prompt, next_token], axis=1)

        # 2. Method A: Full-sequence prefill
        logits_full = model(full_seq)

        # 3. Method B: Prefill then Step
        cache = MambaCache.from_model(model, batch_size=B)
        _ = model(prompt, cache=cache)

        # Run step recurrence
        logits_step = model.step(next_token, cache=cache)

        # 4. Compare outputs
        np_full = np.array(logits_full[:, 4, :])
        np_step = np.array(logits_step)

        diff = np.abs(np_full - np_step).max()
        print(f"MIMO prefill-step max diff: {diff}")
        self.assertTrue(diff < 1e-5, f"MIMO prefill-step mismatch: {diff}")


if __name__ == "__main__":
    unittest.main()
