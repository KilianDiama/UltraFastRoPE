🚀 SovereignEngineV14 — Ultra‑Fast RoPE Transformer Block
SovereignEngineV14 is a modern, production‑grade Transformer block featuring:

UltraFastRoPE (high‑performance Rotary Positional Embeddings)

Grouped Query Attention (GQA) with efficient KV‑head expansion

SwiGLU MLP with fused projections

LayerScale on both attention and MLP paths

PyTorch native SDPA (scaled_dot_product_attention)

Memory‑efficient tensor layouts

KV‑cache compatible via seq_offset

This module is designed for speed, numerical stability, and easy integration into custom Transformer architectures.

✨ Features
🔹 UltraFastRoPE
Precomputed cos/sin tables in float32

Interleaved rotation: (x0, x1) → (-x1, x0)

No complex dtype

Broadcast‑friendly layout [B, H, L, D]

KV‑cache support via seq_offset

🔹 SovereignEngineV14 Transformer Block
GQA: n_heads can differ from n_kv_heads

Fused Q/K/V projections

SwiGLU MLP with fused gate/up projection

LayerScale for deep‑network stability

RMSNorm (fast, stable)

PyTorch SDPA for high‑performance attention

📦 Installation
bash
pip install torch
No external dependencies required.

🧩 Usage Example
python
import torch
from sovereign_engine_v14 import SovereignEngineV14

# Model configuration
d_model = 512
n_heads = 8
n_kv_heads = 2
d_ff = 1408
max_len = 8192

# Instantiate the block
block = SovereignEngineV14(
    d_model=d_model,
    n_heads=n_heads,
    n_kv_heads=n_kv_heads,
    d_ff=d_ff,
    max_len=max_len,
)

# Dummy batch
x = torch.randn(1, 128, d_model)

# Forward pass
out = block(x)
print(out.shape)  # -> [1, 128, 512]
🧠 Technical Overview
🔸 UltraFastRoPE
Precomputes RoPE frequencies as [1, 1, L, D] buffers

Casts once to q.dtype

Interleaved rotation avoids concat/stack overhead

Works for both Q and K tensors

🔸 Attention (GQA)
Q: [B, H, L, D]

K/V: [B, n_kv_heads, L, D]

Efficient expansion using repeat_interleave or broadcast

SDPA handles masking and causal mode

🔸 SwiGLU MLP
gate_up produces [2 * d_ff]

silu(gate) * up fused

down projects back to d_model

🔸 LayerScale
γ₁ and γ₂ initialized based on layer index

Stabilizes deep residual networks

🧪 Causal Mask Example
python
mask = torch.full((1, n_heads, 128, 128), float("-inf"))
mask = torch.triu(mask, diagonal=1)

out = block(x, mask=mask, causal=True)
📁 Project Structure
Code
SovereignEngineV14/
│
├── sovereign_engine_v14.py   # Full implementation
├── README.md                 # This file
└── LICENSE                   # UOSACL‑1.0 license
🔒 License
This project uses the UOSACL‑1.0 — Universal Open‑Source Attribution & Commercial License.

Non‑commercial use: free

Attribution: required

Commercial use: requires agreement + royalties

🧭 Roadmap
[ ] Full KV‑cache integration (external cache tensors)

[ ] FP8 / quantization‑friendly variant

[ ] Multi‑layer SovereignTransformer

[ ] Benchmarks vs LLaMA‑3 / Mistral‑v3

🤝 Contributing
Contributions are welcome — optimizations, CUDA kernels, benchmarks, or architectural improvements.

🔥 Summary
SovereignEngineV14 is a fast, modern, and clean Transformer block suitable for:

LLM research

Custom inference engines

Experimental architectures

IA‑physics hybrid models

High‑performance sequence modeling

It is simple to integrate, stable in training, and optimized for real‑world use.
