import unittest
import math
import numpy as np
import torch
import mlx.core as mx
import mlx.utils as utils

# Import our MLX implementation
from mlx_mamba_native.model import Mamba3 as MlxMamba3, MambaLMHeadModel as MlxMambaLMHeadModel, MambaConfig as MlxMambaConfig

# Import reference PyTorch implementation
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ref_mamba3.mamba3 import Mamba3 as TorchMamba3, MambaLMHeadModel as TorchMambaLMHeadModel, MambaConfig as TorchMambaConfig

def copy_weights_torch_to_mlx(torch_model, mlx_model):
    torch_state_dict = torch_model.state_dict()
    mlx_params = {}
    for k, v in torch_state_dict.items():
        val_np = v.detach().cpu().numpy()
        mlx_params[k] = mx.array(val_np)
    nested_params = utils.tree_unflatten(list(mlx_params.items()))
    mlx_model.update(nested_params)


class TestMamba3Numerical(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        mx.random.seed(42)

    def test_siso_forward(self):
        # 1. Instantiate both models with same configs
        torch_model = TorchMamba3(
            d_model=128,
            d_state=32,
            expand=2,
            headdim=32,
            ngroups=1,
            is_mimo=False
        )
        
        mlx_model = MlxMamba3(
            d_model=128,
            d_state=32,
            expand=2,
            headdim=32,
            ngroups=1,
            is_mimo=False
        )

        copy_weights_torch_to_mlx(torch_model, mlx_model)

        # 2. Prepare random inputs
        B, L, D = 2, 8, 128
        x_np = np.random.randn(B, L, D).astype(np.float32)
        
        x_torch = torch.tensor(x_np)
        x_mlx = mx.array(x_np)

        # 3. Run forward passes
        y_torch = torch_model(x_torch)
        y_mlx = mlx_model(x_mlx)

        # 4. Compare outputs
        y_torch_np = y_torch.detach().cpu().numpy()
        y_mlx_np = np.array(y_mlx)

        diff = np.abs(y_torch_np - y_mlx_np).max()
        print(f"SISO forward max diff: {diff}")
        self.assertTrue(diff < 1e-5, f"SISO forward difference too large: {diff}")

    def test_mimo_forward(self):
        # 1. Instantiate both models
        torch_model = TorchMamba3(
            d_model=128,
            d_state=32,
            expand=2,
            headdim=32,
            ngroups=1,
            is_mimo=True,
            mimo_rank=2
        )
        
        mlx_model = MlxMamba3(
            d_model=128,
            d_state=32,
            expand=2,
            headdim=32,
            ngroups=1,
            is_mimo=True,
            mimo_rank=2
        )

        copy_weights_torch_to_mlx(torch_model, mlx_model)

        # 2. Inputs
        B, L, D = 2, 8, 128
        x_np = np.random.randn(B, L, D).astype(np.float32)
        
        x_torch = torch.tensor(x_np)
        x_mlx = mx.array(x_np)

        # 3. Run forward
        y_torch = torch_model(x_torch)
        y_mlx = mlx_model(x_mlx)

        # 4. Compare
        y_torch_np = y_torch.detach().cpu().numpy()
        y_mlx_np = np.array(y_mlx)

        diff = np.abs(y_torch_np - y_mlx_np).max()
        print(f"MIMO forward max diff: {diff}")
        self.assertTrue(diff < 1e-5, f"MIMO forward difference too large: {diff}")

    def test_siso_step(self):
        torch_model = TorchMamba3(
            d_model=128,
            d_state=32,
            expand=2,
            headdim=32,
            ngroups=1,
            is_mimo=False
        )
        
        mlx_model = MlxMamba3(
            d_model=128,
            d_state=32,
            expand=2,
            headdim=32,
            ngroups=1,
            is_mimo=False
        )

        copy_weights_torch_to_mlx(torch_model, mlx_model)

        B = 2
        # Allocate caches
        t_angle, t_ssm, t_bx = torch_model.allocate_inference_cache(B)
        m_angle, m_ssm, m_bx = mlx_model.allocate_inference_cache(B)

        # Copy state initializations (though zeros, we copy anyway)
        t_angle = torch.tensor(np.array(m_angle))
        t_ssm = torch.tensor(np.array(m_ssm))
        t_bx = torch.tensor(np.array(m_bx))

        # Single step input
        u_np = np.random.randn(B, 128).astype(np.float32)
        u_torch = torch.tensor(u_np)
        u_mlx = mx.array(u_np)

        t_out, t_angle, t_ssm, t_bx = torch_model.step(u_torch, t_angle, t_ssm, t_bx)
        m_out, m_angle, m_ssm, m_bx = mlx_model.step(u_mlx, m_angle, m_ssm, m_bx)

        t_out_np = t_out.detach().cpu().numpy()
        m_out_np = np.array(m_out)

        diff = np.abs(t_out_np - m_out_np).max()
        print(f"SISO step max diff: {diff}")
        self.assertTrue(diff < 1e-5, f"SISO step output difference too large: {diff}")

    def test_mimo_step(self):
        torch_model = TorchMamba3(
            d_model=128,
            d_state=32,
            expand=2,
            headdim=32,
            ngroups=1,
            is_mimo=True,
            mimo_rank=2
        )
        
        mlx_model = MlxMamba3(
            d_model=128,
            d_state=32,
            expand=2,
            headdim=32,
            ngroups=1,
            is_mimo=True,
            mimo_rank=2
        )

        copy_weights_torch_to_mlx(torch_model, mlx_model)

        B = 2
        t_angle, t_ssm, t_bx = torch_model.allocate_inference_cache(B)
        m_angle, m_ssm, m_bx = mlx_model.allocate_inference_cache(B)

        u_np = np.random.randn(B, 128).astype(np.float32)
        u_torch = torch.tensor(u_np)
        u_mlx = mx.array(u_np)

        t_out, t_angle, t_ssm, t_bx = torch_model.step(u_torch, t_angle, t_ssm, t_bx)
        m_out, m_angle, m_ssm, m_bx = mlx_model.step(u_mlx, m_angle, m_ssm, m_bx)

        t_out_np = t_out.detach().cpu().numpy()
        m_out_np = np.array(m_out)

        diff = np.abs(t_out_np - m_out_np).max()
        print(f"MIMO step max diff: {diff}")
        self.assertTrue(diff < 1e-5, f"MIMO step output difference too large: {diff}")


if __name__ == "__main__":
    unittest.main()
