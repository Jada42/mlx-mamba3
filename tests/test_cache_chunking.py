import unittest
import numpy as np
import mlx.core as mx

from mlx_mamba_native.cache import MambaCache
from mlx_mamba_native.model import MambaConfig, MambaLMHeadModel


class TestCacheChunking(unittest.TestCase):

    def setUp(self):
        mx.random.seed(42)

    def assert_chunked_prefill_matches_full(self, config):
        model = MambaLMHeadModel(config)
        tokens = mx.array([[1, 2, 3, 4, 5, 6]])

        logits_full = model(tokens)

        cache = MambaCache.from_model(model, batch_size=1)
        _ = model(tokens[:, :3], cache=cache)
        logits_chunk = model(tokens[:, 3:], cache=cache)

        full_tail = np.array(logits_full[:, 3:, :])
        chunk_tail = np.array(logits_chunk)
        diff = np.abs(full_tail - chunk_tail).max()
        self.assertTrue(diff < 1e-5, f"chunked prefill mismatch: {diff}")

    def test_siso_chunked_prefill(self):
        self.assert_chunked_prefill_matches_full(
            MambaConfig(
                d_model=64,
                n_layer=2,
                vocab_size=128,
                ssm_cfg={"headdim": 32, "is_mimo": False},
            )
        )

    def test_mimo_chunked_prefill(self):
        self.assert_chunked_prefill_matches_full(
            MambaConfig(
                d_model=64,
                n_layer=2,
                vocab_size=128,
                ssm_cfg={"headdim": 32, "is_mimo": True, "mimo_rank": 2},
            )
        )

    def test_hybrid_chunked_prefill(self):
        self.assert_chunked_prefill_matches_full(
            MambaConfig(
                d_model=64,
                n_layer=2,
                vocab_size=128,
                ssm_cfg={"headdim": 32},
                attn_layer_idx=[1],
                attn_cfg={"num_heads": 2},
            )
        )

    def test_intermediate_ngroups(self):
        for ngroups in (2, 4):
            with self.subTest(ngroups=ngroups):
                model = MambaLMHeadModel(
                    MambaConfig(
                        d_model=128,
                        n_layer=1,
                        vocab_size=128,
                        ssm_cfg={"headdim": 32, "ngroups": ngroups},
                    )
                )
                tokens = mx.array([[1, 2, 3]])
                logits = model(tokens)
                self.assertEqual(logits.shape, (1, 3, model.vocab_size))


if __name__ == "__main__":
    unittest.main()
