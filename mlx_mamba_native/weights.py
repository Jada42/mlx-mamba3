import mlx.core as mx
import mlx.utils as utils
import numpy as np

def remap_state_dict(torch_state_dict):
    """
    Convert a PyTorch state dict (as a dictionary of numpy arrays or torch tensors)
    to a dictionary of MLX arrays.
    """
    mlx_state_dict = {}
    for key, value in torch_state_dict.items():
        # Handle PyTorch tensor or NumPy array
        if hasattr(value, "numpy"):
            val_np = value.detach().cpu().numpy()
        elif isinstance(value, np.ndarray):
            val_np = value
        else:
            val_np = np.array(value)

        # Convert to MLX array
        val_mx = mx.array(val_np)
        
        # In our implementation, since the parameter names and structures are identical
        # to the reference PyTorch implementation, we can use the keys directly.
        mlx_state_dict[key] = val_mx
        
    return mlx_state_dict


def load_weights(model, path: str):
    """
    Load weights from a safetensors file into the model.
    """
    if path.endswith(".safetensors"):
        try:
            # First try loading directly as MLX safetensors
            flat_params = mx.load_safetensors(path)
        except Exception:
            # Fallback to safetensors.numpy in case it's a PyTorch-exported safetensors file
            import safetensors.numpy
            flat_params = safetensors.numpy.load_file(path)
            flat_params = {k: mx.array(v) for k, v in flat_params.items()}
    else:
        raise ValueError("Only .safetensors files are supported")

    nested_params = utils.tree_unflatten(list(flat_params.items()))
    model.update(nested_params)


def save_weights(model, path: str):
    """
    Save the model's weights to a safetensors file.
    """
    flat_params = dict(utils.tree_flatten(model.parameters()))
    mx.save_safetensors(path, flat_params)
