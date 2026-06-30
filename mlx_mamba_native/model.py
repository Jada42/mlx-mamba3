import math
from dataclasses import dataclass, field
import mlx.core as mx
import mlx.nn as nn

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Standard Root Mean Square Layer Normalization in MLX."""

    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dims,))

    def __call__(self, x: mx.array) -> mx.array:
        variance = mx.mean(mx.square(x), axis=-1, keepdims=True)
        return (x / mx.sqrt(variance + self.eps)) * self.weight


def split_by_sizes(x, sizes, axis=-1):
    """Split an array along the specified axis by sizes."""
    indices = []
    curr = 0
    for s in sizes[:-1]:
        curr += s
        indices.append(curr)
    return mx.split(x, indices, axis=axis)


def expand_bc_groups(x, nheads: int, axis: int):
    """Expand B/C group projections to per-head projections."""
    groups = x.shape[axis]
    if groups == nheads:
        return x
    assert nheads % groups == 0, "ngroups must divide nheads"
    return mx.repeat(x, nheads // groups, axis=axis)


def build_rope_freqs(num_angles: int) -> mx.array:
    """Build the standard RoPE inverse-frequency vector in MLX."""
    i = mx.arange(num_angles, dtype=mx.float32)
    freqs = 1.0 / (10000.0 ** (i / num_angles))
    return freqs


def apply_rope(x: mx.array, angles: mx.array) -> mx.array:
    """Rotate pairs of dimensions of x by the given angles in MLX."""
    cos = mx.cos(angles)
    sin = mx.sin(angles)

    x1 = x[..., 0::2]
    x2 = x[..., 1::2]

    x_rotated_1 = x1 * cos - x2 * sin
    x_rotated_2 = x1 * sin + x2 * cos

    out = mx.stack([x_rotated_1, x_rotated_2], axis=-1)
    shape = out.shape[:-2] + (out.shape[-2] * out.shape[-1],)
    return out.reshape(shape)


def associative_scan_hs(fn, x, axis=1):
    """Hillis-Steele parallel prefix scan in pure MLX."""
    a, b = x
    L = a.shape[axis]
    num_steps = int(math.ceil(math.log2(L)))

    for step in range(num_steps):
        gap = 1 << step
        a_left = a[:, :-gap]
        b_left = b[:, :-gap]
        a_right = a[:, gap:]
        b_right = b[:, gap:]

        a_comb, b_comb = fn((a_left, b_left), (a_right, b_right))

        a = mx.concatenate([a[:, :gap], a_comb], axis=axis)
        b = mx.concatenate([b[:, :gap], b_comb], axis=axis)

    return a, b


def mamba_combine(left, right):
    a_l, b_l = left
    a_r, b_r = right
    return a_l * a_r, a_r * b_l + b_r


def mamba3_siso_scan_parallel(
    x,
    B_proj,
    C_proj,
    ADT,
    DT,
    trap,
    D_skip,
    initial_state=None,
    initial_Bx_prev=None,
):
    """SISO Mamba-3 scan using parallel prefix scan."""
    B_batch, L, H, P = x.shape
    D_state = B_proj.shape[-1]

    # Compute outer product contribution B*x: (B, L, H, P, D)
    Bx_curr = mx.einsum("blhp,blhd->blhpd", x, B_proj)

    # Prepare decay before the trapezoidal blend: the previous endpoint is
    # transported through the same exponential transition as h_{t-1}.
    decay = mx.exp(ADT).reshape(B_batch, L, H, 1, 1)

    # Shift Bx_curr by 1 along L dimension to get Bx_prev
    if initial_Bx_prev is None:
        Bx_prev_first = mx.zeros((B_batch, 1, H, P, D_state))
    else:
        Bx_prev_first = mx.expand_dims(initial_Bx_prev, axis=1)
    Bx_prev = mx.concatenate([Bx_prev_first, Bx_curr[:, :-1]], axis=1)

    # Blend current and previous contribution using trapezoidal sigmoid gate
    trap_exp = trap.reshape(B_batch, L, H, 1, 1)
    Bx_blended = (1.0 - trap_exp) * Bx_curr + trap_exp * 0.5 * (Bx_curr + decay * Bx_prev)

    # Prepare decay and input terms for associative scan
    DT_exp = DT.reshape(B_batch, L, H, 1, 1)
    input_term = DT_exp * Bx_blended

    # Run parallel prefix scan
    prefix_decay, h = associative_scan_hs(mamba_combine, (decay, input_term), axis=1)
    if initial_state is not None:
        h = h + prefix_decay * mx.expand_dims(initial_state, axis=1)

    # Output projection contraction and skip connection
    y = mx.einsum("blhd,blhpd->blhp", C_proj, h)
    D_exp = D_skip.reshape(1, 1, H, 1)
    y = y + D_exp * x

    return y, h, Bx_curr


def mamba3_mimo_scan_parallel(
    x,
    B_proj,
    C_proj,
    ADT,
    DT,
    trap,
    D_skip,
    mimo_x,
    mimo_o,
    initial_state=None,
    initial_Bx_prev=None,
):
    """MIMO Mamba-3 scan using parallel prefix scan."""
    B_batch, L, H, P = x.shape
    D_state = B_proj.shape[-1]

    # Project x from headdim P to R rank-scalars per head: (B, L, H, R)
    x_r = mx.einsum("blhp,hrp->blhr", x, mimo_x)

    # Sum of rank-1 contributions: (B, L, H, D)
    Bx_curr = mx.einsum("blhr,blrhd->blhd", x_r, B_proj)

    # Prepare decay before the trapezoidal blend.
    decay = mx.exp(ADT).reshape(B_batch, L, H, 1)

    # Shift Bx_curr by 1 to get Bx_prev
    if initial_Bx_prev is None:
        Bx_prev_first = mx.zeros((B_batch, 1, H, D_state))
    else:
        Bx_prev_first = mx.expand_dims(initial_Bx_prev, axis=1)
    Bx_prev = mx.concatenate([Bx_prev_first, Bx_curr[:, :-1]], axis=1)

    # Blend using trapezoidal sigmoid gate
    trap_exp = trap.reshape(B_batch, L, H, 1)
    Bx_blended = (1.0 - trap_exp) * Bx_curr + trap_exp * 0.5 * (Bx_curr + decay * Bx_prev)

    # Prepare decay and input terms
    DT_exp = DT.reshape(B_batch, L, H, 1)
    input_term = DT_exp * Bx_blended

    # Run parallel prefix scan
    prefix_decay, h = associative_scan_hs(mamba_combine, (decay, input_term), axis=1)
    if initial_state is not None:
        h = h + prefix_decay * mx.expand_dims(initial_state, axis=1)

    # Output computation: y_r_scalar (B, L, R, H)
    y_r_scalar = mx.einsum("blrhd,blhd->blrh", C_proj, h)

    # Skip connection (B, L, R, H)
    skip = D_skip.reshape(1, 1, 1, H) * mx.transpose(x_r, (0, 1, 3, 2))

    y_pre = y_r_scalar + skip

    # Project back to headdim P: (B, L, H, P)
    y = mx.einsum("blrh,hrp->blhp", y_pre, mimo_o)

    return y, h, Bx_curr


# ---------------------------------------------------------------------------
# Mamba3 Module
# ---------------------------------------------------------------------------

class Mamba3(nn.Module):
    """Mamba-3 sequence mixing layer in MLX."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int = 1,
        rope_fraction: float = 0.5,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        A_floor: float = 1e-4,
        is_mimo: bool = False,
        mimo_rank: int = 4,
    ):
        super().__init__()

        self.d_model   = d_model
        self.d_state   = d_state
        self.expand    = expand
        self.headdim   = headdim
        self.A_floor   = A_floor
        self.is_mimo   = is_mimo
        self.mimo_rank = mimo_rank if is_mimo else 1
        self.num_bc_heads = ngroups

        self.d_inner = int(expand * d_model)
        assert self.d_inner % headdim == 0, "d_inner must be divisible by headdim"
        self.nheads = self.d_inner // headdim
        assert ngroups > 0 and self.nheads % ngroups == 0, "ngroups must divide nheads"

        assert rope_fraction in [0.5, 1.0], "Only rope_fraction ∈ {0.5, 1.0} supported"
        self.split_tensor_size = int(d_state * rope_fraction)
        if self.split_tensor_size % 2 != 0:
            self.split_tensor_size -= 1
        self.num_rope_angles = self.split_tensor_size // 2
        assert self.num_rope_angles > 0

        # Input projection dimension
        d_in_proj = (
            2 * self.d_inner
            + 2 * d_state * ngroups * self.mimo_rank
            + 3 * self.nheads
            + self.num_rope_angles
        )
        self.in_proj = nn.Linear(d_model, d_in_proj, bias=False)

        # dt bias setup
        _dt = mx.maximum(
            mx.exp(
                mx.random.uniform(shape=(self.nheads,))
                * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ),
            dt_init_floor
        )
        _dt_bias = _dt + mx.log(-mx.expm1(-_dt))
        self.dt_bias = _dt_bias

        # B and C biases
        self.B_bias = mx.ones((self.nheads, self.mimo_rank, d_state))
        self.C_bias = mx.ones((self.nheads, self.mimo_rank, d_state))

        # RMS norms for B and C
        self.B_norm = RMSNorm(d_state)
        self.C_norm = RMSNorm(d_state)

        # MIMO projection matrices
        if self.is_mimo:
            self.mimo_x = mx.ones((self.nheads, self.mimo_rank, self.headdim)) / self.mimo_rank
            self.mimo_z = mx.ones((self.nheads, self.mimo_rank, self.headdim))
            self.mimo_o = mx.ones((self.nheads, self.mimo_rank, self.headdim)) / self.mimo_rank

        # D skip connection
        self.D = mx.ones((self.nheads,))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def __call__(self, u: mx.array, cache=None, layer_idx=None) -> mx.array:
        """
        u: (batch, seq_len, d_model)
        """
        batch, L, _ = u.shape

        # Step 1: Single fused projection
        zxBCdtAtrap = self.in_proj(u)

        # Step 2: Split into named components
        sizes = [
            self.d_inner,
            self.d_inner,
            self.d_state * self.num_bc_heads * self.mimo_rank,
            self.d_state * self.num_bc_heads * self.mimo_rank,
            self.nheads,
            self.nheads,
            self.nheads,
            self.num_rope_angles,
        ]
        (z, x, B_raw, C_raw,
         dd_dt, dd_A, trap_raw, angle_raw) = split_by_sizes(zxBCdtAtrap, sizes, axis=-1)

        # Reshapes
        z = z.reshape(batch, L, self.nheads, self.headdim)
        x = x.reshape(batch, L, self.nheads, self.headdim)

        B_raw = B_raw.reshape(batch, L, self.mimo_rank, self.num_bc_heads, self.d_state)
        C_raw = C_raw.reshape(batch, L, self.mimo_rank, self.num_bc_heads, self.d_state)

        # Step 3: Compute state decay and time step
        A = -nn.softplus(dd_A)
        A = mx.minimum(A, -self.A_floor)
        DT = nn.softplus(dd_dt + self.dt_bias)
        ADT = A * DT

        # Step 4: Trapezoidal gate
        trap = mx.sigmoid(trap_raw)

        # Step 5: Normalize and expand groups to heads
        B_normed = self.B_norm(B_raw)
        C_normed = self.C_norm(C_raw)

        B_exp = expand_bc_groups(B_normed, self.nheads, axis=3)
        C_exp = expand_bc_groups(C_normed, self.nheads, axis=3)

        # Add biases
        B_bias_t = mx.transpose(self.B_bias, (1, 0, 2))  # (R, H, D)
        C_bias_t = mx.transpose(self.C_bias, (1, 0, 2))  # (R, H, D)
        B_exp = B_exp + B_bias_t
        C_exp = C_exp + C_bias_t

        # Step 6: Apply RoPE rotation
        initial_angle_state = None
        initial_ssm_state = None
        initial_Bx_prev_state = None
        if cache is not None and layer_idx is not None:
            initial_angle_state, initial_ssm_state, initial_Bx_prev_state = cache.get_layer(layer_idx)

        angle_increments = mx.expand_dims(angle_raw, 2) * mx.expand_dims(DT, -1)  # (B, L, 1, S) * (B, L, H, 1) -> (B, L, H, S)
        cumulative_angles = mx.cumsum(angle_increments, axis=1)
        if initial_angle_state is not None:
            cumulative_angles = cumulative_angles + mx.expand_dims(initial_angle_state, axis=1)

        angles_for_rot = mx.broadcast_to(
            mx.expand_dims(cumulative_angles, 2),
            (batch, L, self.mimo_rank, self.nheads, self.num_rope_angles)
        )

        B_rot = apply_rope(B_exp[..., :self.split_tensor_size], angles_for_rot)
        C_rot = apply_rope(C_exp[..., :self.split_tensor_size], angles_for_rot)

        B_proj = mx.concatenate([B_rot, B_exp[..., self.split_tensor_size:]], axis=-1)
        C_proj = mx.concatenate([C_rot, C_exp[..., self.split_tensor_size:]], axis=-1)

        # Step 7: Parallel SSM scan
        if self.is_mimo:
            y, h, Bx_curr = mamba3_mimo_scan_parallel(
                x=x,
                B_proj=B_proj,
                C_proj=C_proj,
                ADT=ADT,
                DT=DT,
                trap=trap,
                D_skip=self.D,
                mimo_x=self.mimo_x,
                mimo_o=self.mimo_o,
                initial_state=initial_ssm_state,
                initial_Bx_prev=initial_Bx_prev_state,
            )
            # Gate output with z using simple SiLU
            y = y * nn.silu(z)
        else:
            y, h, Bx_curr = mamba3_siso_scan_parallel(
                x=x,
                B_proj=B_proj[:, :, 0],
                C_proj=C_proj[:, :, 0],
                ADT=ADT,
                DT=DT,
                trap=trap,
                D_skip=self.D,
                initial_state=initial_ssm_state,
                initial_Bx_prev=initial_Bx_prev_state,
            )
            # Gate output with z using simple SiLU
            y = y * nn.silu(z)

        # If cache is provided (prefill path), extract final states and populate cache
        if cache is not None and layer_idx is not None:
            angle_state = cumulative_angles[:, -1]
            ssm_state = h[:, -1]
            Bx_prev_state = Bx_curr[:, -1]
            cache.update_layer(layer_idx, angle_state, ssm_state, Bx_prev_state)

        # Step 8: Output projection
        y = y.reshape(batch, L, -1)
        out = self.out_proj(y)
        return out

    def step(self, u, angle_state, ssm_state, Bx_prev_state):
        """Run a single autoregressive step."""
        batch = u.shape[0]

        # In-projection
        zxBCdtAtrap = self.in_proj(u)

        sizes = [
            self.d_inner,
            self.d_inner,
            self.d_state * self.num_bc_heads * self.mimo_rank,
            self.d_state * self.num_bc_heads * self.mimo_rank,
            self.nheads,
            self.nheads,
            self.nheads,
            self.num_rope_angles,
        ]
        (z, x, B_raw, C_raw,
         dd_dt, dd_A, trap_raw, angle_raw) = split_by_sizes(zxBCdtAtrap, sizes, axis=-1)

        # Reshapes
        z = z.reshape(batch, self.nheads, self.headdim)
        x = x.reshape(batch, self.nheads, self.headdim)

        B_raw = B_raw.reshape(batch, self.mimo_rank, self.num_bc_heads, self.d_state)
        C_raw = C_raw.reshape(batch, self.mimo_rank, self.num_bc_heads, self.d_state)

        # Discretization parameters
        A = -nn.softplus(dd_A)
        A = mx.minimum(A, -self.A_floor)
        DT = nn.softplus(dd_dt + self.dt_bias)
        ADT = A * DT
        trap = mx.sigmoid(trap_raw)

        # RMS normalization and head expansion
        B_normed = self.B_norm(B_raw)
        C_normed = self.C_norm(C_raw)

        B_exp = expand_bc_groups(B_normed, self.nheads, axis=2)
        C_exp = expand_bc_groups(C_normed, self.nheads, axis=2)

        # Add biases
        B_bias_t = mx.transpose(self.B_bias, (1, 0, 2))
        C_bias_t = mx.transpose(self.C_bias, (1, 0, 2))
        B_exp = B_exp + B_bias_t
        C_exp = C_exp + C_bias_t

        # RoPE updates
        delta_angle = mx.expand_dims(angle_raw, 1) * mx.expand_dims(DT, -1)
        angle_state = angle_state + delta_angle

        angles_for_rot = mx.broadcast_to(
            mx.expand_dims(angle_state, 1),
            (batch, self.mimo_rank, self.nheads, self.num_rope_angles)
        )

        B_rot = apply_rope(B_exp[..., :self.split_tensor_size], angles_for_rot)
        C_rot = apply_rope(C_exp[..., :self.split_tensor_size], angles_for_rot)

        B_proj = mx.concatenate([B_rot, B_exp[..., self.split_tensor_size:]], axis=-1)
        C_proj = mx.concatenate([C_rot, C_exp[..., self.split_tensor_size:]], axis=-1)

        # SSM update step
        decay = mx.exp(ADT)
        tr = trap

        if self.is_mimo:
            # MIMO step
            x_r = mx.einsum("bhp,hrp->bhr", x, self.mimo_x)
            Bx_curr = mx.einsum("bhr,brhd->bhd", x_r, B_proj)

            tr_e = mx.expand_dims(tr, -1)
            Bx_blended = (1.0 - tr_e) * Bx_curr + tr_e * 0.5 * (Bx_curr + mx.expand_dims(decay, -1) * Bx_prev_state)
            ssm_state = mx.expand_dims(decay, -1) * ssm_state + mx.expand_dims(DT, -1) * Bx_blended

            y_r_scalar = mx.einsum("brhd,bhd->brh", C_proj, ssm_state)
            skip = self.D.reshape(1, 1, self.nheads) * mx.transpose(x_r, (0, 2, 1))
            y_pre = y_r_scalar + skip
            y = mx.einsum("brh,hrp->bhp", y_pre, self.mimo_o)
            y = y * nn.silu(z)

            Bx_prev_state = Bx_curr
        else:
            # SISO step
            Bx_curr = mx.einsum("bhp,bhd->bhpd", x, B_proj[:, 0])
            tr_e = mx.expand_dims(mx.expand_dims(tr, -1), -1)
            Bx_blended = (1.0 - tr_e) * Bx_curr + tr_e * 0.5 * (Bx_curr + mx.expand_dims(mx.expand_dims(decay, -1), -1) * Bx_prev_state)
            ssm_state = mx.expand_dims(mx.expand_dims(decay, -1), -1) * ssm_state + mx.expand_dims(mx.expand_dims(DT, -1), -1) * Bx_blended

            y = mx.einsum("bhd,bhpd->bhp", C_proj[:, 0], ssm_state)
            y = y + self.D.reshape(1, self.nheads, 1) * x
            y = y * nn.silu(z)

            Bx_prev_state = Bx_curr

        y = y.reshape(batch, -1)
        out = self.out_proj(y)
        return out, angle_state, ssm_state, Bx_prev_state

    def allocate_inference_cache(self, batch_size: int):
        """Allocate zero-initialized cache tensors."""
        angle_state = mx.zeros((batch_size, self.nheads, self.num_rope_angles))

        if self.is_mimo:
            ssm_state = mx.zeros((batch_size, self.nheads, self.d_state))
            Bx_prev_state = mx.zeros((batch_size, self.nheads, self.d_state))
        else:
            ssm_state = mx.zeros((batch_size, self.nheads, self.headdim, self.d_state))
            Bx_prev_state = mx.zeros((batch_size, self.nheads, self.headdim, self.d_state))

        return angle_state, ssm_state, Bx_prev_state


# ---------------------------------------------------------------------------
# Stacked Model Structure
# ---------------------------------------------------------------------------

@dataclass
class MambaConfig:
    d_model: int = 256
    d_intermediate: int = 0
    n_layer: int = 4
    vocab_size: int = 50277
    ssm_cfg: dict = field(default_factory=dict)
    attn_layer_idx: list = field(default_factory=list)
    attn_cfg: dict = field(default_factory=dict)
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 8
    tie_embeddings: bool = True


class MambaBlock(nn.Module):
    """Single Mamba-3 block: Norm → Mamba3 → residual add."""

    def __init__(self, d_model: int, ssm_cfg: dict):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.mixer = Mamba3(d_model=d_model, **ssm_cfg)

    def __call__(self, x: mx.array, cache=None, layer_idx=None) -> mx.array:
        return x + self.mixer(self.norm(x), cache=cache, layer_idx=layer_idx)


class CausalAttention(nn.Module):
    """Causal self-attention layer with KV cache support."""

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def __call__(self, x: mx.array, cache=None, layer_idx=None) -> mx.array:
        B, L, D = x.shape
        Q = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = self.k_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = self.v_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        past_len = 0
        if cache is not None and layer_idx is not None:
            K_cached, V_cached, _ = cache.get_layer(layer_idx)
            past_len = K_cached.shape[2]
            K = mx.concatenate([K_cached, K], axis=2)
            V = mx.concatenate([V_cached, V], axis=2)
            cache.update_layer(layer_idx, K, V, None)

        scores = (Q @ K.transpose(0, 1, 3, 2)) / math.sqrt(self.head_dim)
        total_len = K.shape[2]
        q_idx = mx.arange(L).reshape(L, 1)
        k_idx = mx.arange(total_len).reshape(1, total_len)
        mask = mx.where(k_idx <= past_len + q_idx, 0.0, -1e9)
        mask = mask.reshape(1, 1, L, total_len)
        scores = scores + mask
        attn = mx.softmax(scores, axis=-1)
        out = attn @ V
        out = out.transpose(0, 2, 1, 3).reshape(B, L, D)
        return self.out_proj(out)

    def step(self, x: mx.array, K_cached: mx.array, V_cached: mx.array):
        B, D = x.shape
        Q = self.q_proj(x).reshape(B, 1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K_new = self.k_proj(x).reshape(B, 1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V_new = self.v_proj(x).reshape(B, 1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        K_cached = mx.concatenate([K_cached, K_new], axis=2)
        V_cached = mx.concatenate([V_cached, V_new], axis=2)

        scores = (Q @ K_cached.transpose(0, 1, 3, 2)) / math.sqrt(self.head_dim)
        attn = mx.softmax(scores, axis=-1)
        out = attn @ V_cached
        out = out.transpose(0, 2, 1, 3).reshape(B, D)
        return self.out_proj(out), K_cached, V_cached


class AttentionBlock(nn.Module):
    """Single Self-Attention block: Norm → CausalAttention → residual add."""

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.attn = CausalAttention(d_model, num_heads)

    def __call__(self, x: mx.array, cache=None, layer_idx=None) -> mx.array:
        return x + self.attn(self.norm(x), cache=cache, layer_idx=layer_idx)


class MLP(nn.Module):
    """SwiGLU-style Feed-Forward Network."""

    def __init__(self, d_model: int, d_intermediate: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, 2 * d_intermediate, bias=False)
        self.fc2 = nn.Linear(d_intermediate, d_model, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        gate, val = mx.split(self.fc1(x), 2, axis=-1)
        return self.fc2(nn.silu(gate) * val)


class MambaLMHeadModel(nn.Module):
    """Mamba-3 language model with optional CausalAttention layers."""

    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config

        vocab_size = config.vocab_size
        r = vocab_size % config.pad_vocab_size_multiple
        if r != 0:
            vocab_size += config.pad_vocab_size_multiple - r
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, config.d_model)

        self.layers = []
        for i in range(config.n_layer):
            if i in config.attn_layer_idx:
                num_heads = config.attn_cfg.get("num_heads", 4)
                self.layers.append(AttentionBlock(config.d_model, num_heads))
            else:
                self.layers.append(MambaBlock(config.d_model, config.ssm_cfg))

        if config.d_intermediate > 0:
            self.mlp_norms = [RMSNorm(config.d_model) for _ in range(config.n_layer)]
            self.mlp_layers = [
                MLP(config.d_model, config.d_intermediate)
                for _ in range(config.n_layer)
            ]
        else:
            self.mlp_norms = None
            self.mlp_layers = None

        self.norm_f = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, vocab_size, bias=False)

        if config.tie_embeddings:
            self.lm_head.weight = self.embedding.weight

    def __call__(self, input_ids: mx.array, cache=None) -> mx.array:
        x = self.embedding(input_ids)
        for i, block in enumerate(self.layers):
            x = block(x, cache=cache, layer_idx=i)
            if self.mlp_layers is not None:
                x = x + self.mlp_layers[i](self.mlp_norms[i](x))
        x = self.norm_f(x)
        return self.lm_head(x)

    def step(self, input_ids: mx.array, cache):
        """Run a single step of autoregressive decoding."""
        if len(input_ids.shape) > 1 and input_ids.shape[1] == 1:
            input_ids = input_ids.squeeze(1)

        x = self.embedding(input_ids)

        for i, block in enumerate(self.layers):
            if isinstance(block, MambaBlock):
                angle_s, h_s, bx_s = cache.get_layer(i)
                x_norm = block.norm(x)
                out, angle_s, h_s, bx_s = block.mixer.step(x_norm, angle_s, h_s, bx_s)
                cache.update_layer(i, angle_s, h_s, bx_s)
                x = x + out
            elif isinstance(block, AttentionBlock):
                K_cache, V_cache, _ = cache.get_layer(i)
                x_norm = block.norm(x)
                out, K_cache, V_cache = block.attn.step(x_norm, K_cache, V_cache)
                cache.update_layer(i, K_cache, V_cache, None)
                x = x + out

            if self.mlp_layers is not None:
                x = x + self.mlp_layers[i](self.mlp_norms[i](x))

        x = self.norm_f(x)
        logits = self.lm_head(x)
        return logits
