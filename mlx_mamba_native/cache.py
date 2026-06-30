import mlx.core as mx

class MambaCacheState:
    """State holder for a single Mamba-3 layer."""
    def __init__(self, angle_state: mx.array, ssm_state: mx.array, Bx_prev_state: mx.array):
        self.angle_state = angle_state
        self.ssm_state = ssm_state
        self.Bx_prev_state = Bx_prev_state


class MambaCache:
    """SSM cache container analogous to KVCache in transformers/mlx-lm."""
    def __init__(self, states=None):
        self.states = states or []

    @classmethod
    def from_model(cls, model, batch_size: int = 1):
        states = []
        for layer in model.layers:
            name = layer.__class__.__name__
            if name == "MambaBlock":
                angle_s, h_s, bx_s = layer.mixer.allocate_inference_cache(batch_size)
                states.append(MambaCacheState(angle_s, h_s, bx_s))
            elif name == "AttentionBlock":
                K_cache = mx.zeros((batch_size, layer.attn.num_heads, 0, layer.attn.head_dim))
                V_cache = mx.zeros((batch_size, layer.attn.num_heads, 0, layer.attn.head_dim))
                states.append(MambaCacheState(K_cache, V_cache, None))
            else:
                raise ValueError(f"Unknown layer type: {name}")
        return cls(states)

    def update_layer(self, layer_idx: int, angle_state: mx.array, ssm_state: mx.array, Bx_prev_state: mx.array):
        self.states[layer_idx].angle_state = angle_state
        self.states[layer_idx].ssm_state = ssm_state
        self.states[layer_idx].Bx_prev_state = Bx_prev_state

    def get_layer(self, layer_idx: int):
        state = self.states[layer_idx]
        return state.angle_state, state.ssm_state, state.Bx_prev_state
