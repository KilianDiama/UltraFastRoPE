import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class UltraFastRoPE(nn.Module):
    """
    Ultra-fast RoPE with interleaved rotation.

    Design choices:
    - Precomputes cos/sin on [max_seq_len, dim] in float32 for numerical stability.
    - Stores tables as buffers on the target device; forward casts once to q.dtype.
    - Interleaved rotation: (x0, x1) -> (-x1, x0) without stack/concat.
    - Layout compatible with [B, H, L, D] tensors.

    Args:
        dim: Head dimension (D).
        max_seq_len: Maximum sequence length supported.
        base: RoPE base frequency.
        device: Optional device for buffer allocation.
        dtype: Ignored for internal tables (forced to float32), kept for API symmetry.
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 8192,
        base: float = 10000.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,  # kept for API symmetry
    ) -> None:
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        # Tables are always float32 for stability; we cast to q.dtype in forward.
        factory_kwargs = {"device": device, "dtype": torch.float32}

        if dim % 2 != 0:
            raise ValueError(f"RoPE dim must be even, got dim={dim}")

        # inv_freq: [dim/2]
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, **factory_kwargs) / dim))
        # t: [L]
        t = torch.arange(max_seq_len, **factory_kwargs)
        # freqs: [L, dim/2]
        freqs = torch.outer(t, inv_freq)  # [L, dim/2]

        # Interleaved format: duplicate freqs along last dim to match [L, dim]
        emb = torch.cat([freqs, freqs], dim=-1)  # [L, dim]

        # [1, 1, L, dim] for easy broadcasting over [B, H, L, D]
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]

        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    @staticmethod
    def _rotate_interleaved(x: torch.Tensor) -> torch.Tensor:
        """
        Rotate pairs in interleaved layout: (x0, x1) -> (-x1, x0).

        Args:
            x: [..., dim] with dim even.

        Returns:
            Rotated tensor with same shape as x.
        """
        x0 = x[..., 0::2]
        x1 = x[..., 1::2]
        out = torch.empty_like(x)
        out[..., 0::2] = -x1
        out[..., 1::2] = x0
        return out

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RoPE to q and k.

        Args:
            q: [B, H, L, D]
            k: [B, H_kv, L, D]
            seq_offset: starting position in the cached RoPE table (for KV cache).

        Returns:
            q_out, k_out: same shapes as q, k.
        """
        if q.size(-1) != self.dim:
            raise ValueError(f"RoPE dim mismatch: got {q.size(-1)}, expected {self.dim}")
        if k.size(-1) != self.dim:
            raise ValueError(f"RoPE dim mismatch for k: got {k.size(-1)}, expected {self.dim}")

        bsz, n_heads, L, D = q.shape
        end_pos = seq_offset + L
        if end_pos > self.max_seq_len:
            raise ValueError(
                f"Requested positions [{seq_offset}, {end_pos}) exceed "
                f"max_seq_len={self.max_seq_len}. Increase max_seq_len in UltraFastRoPE."
            )

        # cos/sin: [1, 1, L, D] -> broadcast over [B, H, L, D]
        # Single cast to q.dtype for both q and k.
        cos = self.cos_cached[:, :, seq_offset:end_pos, :].to(q.dtype)
        sin = self.sin_cached[:, :, seq_offset:end_pos, :].to(q.dtype)

        q_rot = self._rotate_interleaved(q)
        k_rot = self._rotate_interleaved(k)

        q_out = q * cos + q_rot * sin
        k_out = k * cos + k_rot * sin
        return q_out, k_out


class SovereignEngineV14(nn.Module):
    """
    SovereignEngineV14 Transformer block.

    Features:
    - GQA with n_heads, n_kv_heads.
    - UltraFastRoPE on head_dim.
    - SwiGLU MLP.
    - LayerScale on both attention and MLP paths.
    - Uses PyTorch scaled_dot_product_attention.

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        n_kv_heads: Number of key/value heads (for GQA).
        d_ff: Hidden dimension of MLP.
        max_len: Maximum sequence length for RoPE.
        dropout: Dropout probability.
        layer_idx: Layer index (for LayerScale init).
        device: Optional device for parameters.
        dtype: Optional dtype for parameters.
    """

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        n_kv_heads: int = 2,
        d_ff: int = 1408,
        max_len: int = 8192,
        dropout: float = 0.0,
        layer_idx: int = 0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        if n_heads % n_kv_heads != 0:
            raise ValueError(f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})")

        factory_kwargs = {"device": device, "dtype": dtype}

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.kv_group = n_heads // n_kv_heads

        # Norms
        self.norm1 = nn.RMSNorm(d_model, eps=1e-6, **factory_kwargs)
        self.norm2 = nn.RMSNorm(d_model, eps=1e-6, **factory_kwargs)

        # QKV projections
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False, **factory_kwargs)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False, **factory_kwargs)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False, **factory_kwargs)
        self.out_proj = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)

        # RoPE on head_dim
        self.rope = UltraFastRoPE(
            dim=self.head_dim,
            max_seq_len=max_len,
            device=device,
            dtype=dtype,
        )

        # SwiGLU MLP
        self.gate_up = nn.Linear(d_model, 2 * d_ff, bias=False, **factory_kwargs)
        self.down = nn.Linear(d_ff, d_model, bias=False, **factory_kwargs)

        # LayerScale
        gamma_init = 1e-2 if layer_idx < 12 else 1e-3
        gamma = torch.full((d_model,), gamma_init, **factory_kwargs)
        self.gamma1 = nn.Parameter(gamma.clone())
        self.gamma2 = nn.Parameter(gamma.clone())

        self.dropout = nn.Dropout(dropout)

    def _attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        is_causal: bool,
    ) -> torch.Tensor:
        """
        Wrapper around scaled_dot_product_attention to centralize mask/causal logic.

        Args:
            q, k, v: [B, H, L, D]
            attn_mask: broadcastable to [B, H, L, L] or None.
            is_causal: whether to apply causal masking.

        Returns:
            Attention output [B, H, L, D].
        """
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=is_causal,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        seq_offset: int = 0,
        causal: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass of the Transformer block.

        Args:
            x: [B, L, d_model]
            mask: Optional attention mask, broadcastable to [B, n_heads, L, L].
            seq_offset: Starting position for RoPE (for KV cache usage).
            causal: Whether to apply causal masking in attention.

        Returns:
            Updated hidden states [B, L, d_model].
        """
        # --- Attention path ---
        residual = x
        h = self.norm1(x)
        bsz, q_len, _ = h.shape

        # [B, L, d_model] -> [B, H, L, D]
        q = self.q_proj(h).view(bsz, q_len, self.n_heads, self.head_dim).transpose(1, 2).contiguous()
        k = self.k_proj(h).view(bsz, q_len, self.n_kv_heads, self.head_dim).transpose(1, 2).contiguous()
        v = self.v_proj(h).view(bsz, q_len, self.n_kv_heads, self.head_dim).transpose(1, 2).contiguous()

        # Apply RoPE
        q, k = self.rope(q, k, seq_offset=seq_offset)

        # GQA expansion: [B, n_kv_heads, L, D] -> [B, n_heads, L, D]
        if self.kv_group > 1:
            k = k.repeat_interleave(self.kv_group, dim=1)
            v = v.repeat_interleave(self.kv_group, dim=1)

        attn_out = self._attn(q, k, v, attn_mask=mask, is_causal=causal)

        # [B, H, L, D] -> [B, L, d_model]
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, q_len, self.d_model)

        # Output projection + LayerScale
        x = residual + self.dropout(self.out_proj(attn_out)) * self.gamma1

        # --- MLP path (SwiGLU) ---
        residual = x
        h = self.norm2(x)

        gate_up = self.gate_up(h)  # [B, L, 2*d_ff]
        gate, up = gate_up.chunk(2, dim=-1)
        mlp_hidden = F.silu(gate) * up
        mlp_out = self.down(mlp_hidden)

        x = residual + self.dropout(mlp_out) * self.gamma2

        return x
