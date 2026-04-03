# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Export TurboDiffusion WanModel (DiT backbone) to ONNX format.

Creates an ONNX-exportable wrapper that replaces non-traceable operations
(flash_attn rotary embeddings, distributed attention, amp.autocast, einops,
dynamic tensor creation) with pure PyTorch equivalents that survive
torch.onnx.export tracing.

Usage:
    python export_onnx.py --model Wan2.1-1.3B --dit_path checkpoints/model.pth --output model.onnx

The exported model expects FIXED spatial dimensions (set at export time via
--num_frames, --height, --width). The batch dimension is dynamic.

Inputs:
    - x_B_C_T_H_W: [B, 16, T_latent, H_latent, W_latent] float16
    - timesteps_B_T: [B, 1] float16
    - crossattn_emb: [B, 512, 4096] float16

Output:
    - output: [B, 16, T_latent, H_latent, W_latent] float16
"""

import argparse
import math
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

from rcm.utils.model_utils import load_state_dict
from rcm.networks.wan2pt1 import (
    WanModel,
    WanSelfAttention,
    WanT2VCrossAttention,
    WanI2VCrossAttention,
    WanAttentionBlock,
    Head,
    sinusoidal_embedding_1d,
    T5_CONTEXT_TOKEN_NUMBER,
)


# ---------------------------------------------------------------------------
# Pure PyTorch replacements for non-ONNX-compatible ops
# ---------------------------------------------------------------------------


def onnx_rope_apply(x, freqs):
    """
    Pure PyTorch interleaved rotary position embedding.
    Replaces flash_attn's apply_rotary_emb which is not ONNX-exportable.

    Args:
        x: [B, S, H, D]
        freqs: [S, D//2]
    """
    B, S, H, D = x.shape
    half_d = D // 2

    cos = torch.cos(freqs).to(x.dtype)  # [S, D//2]
    sin = torch.sin(freqs).to(x.dtype)

    # Reshape for interleaved pairs: [B, S, H, D//2, 2]
    x = x.reshape(B, S, H, half_d, 2)
    x0 = x[..., 0]
    x1 = x[..., 1]

    # Broadcast: [S, D//2] -> [1, S, 1, D//2]
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)

    o0 = x0 * cos - x1 * sin
    o1 = x0 * sin + x1 * cos

    return torch.stack([o0, o1], dim=-1).reshape(B, S, H, D)


def onnx_sdpa(q, k, v):
    """
    ONNX-friendly scaled dot-product attention.
    q, k, v: [B, S, H, D] -> transposes to [B, H, S, D] for SDPA.
    Returns: [B, S, H, D]
    """
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    out = F.scaled_dot_product_attention(q, k, v)
    return out.transpose(1, 2).contiguous()


# ---------------------------------------------------------------------------
# Pre-compute RoPE frequencies (outside forward, avoids torch.arange in trace)
# ---------------------------------------------------------------------------


def precompute_rope_freqs(rope_emb, T, H, W):
    """
    Compute RoPE frequency tensor for given spatial/temporal dimensions.
    Returns: [T*H*W, head_dim//2] float32 tensor.
    """
    dim_h = rope_emb._dim_h
    dim_t = rope_emb._dim_t

    seq = torch.arange(max(rope_emb.max_h, rope_emb.max_w, rope_emb.max_t)).float()
    dim_spatial_range = torch.arange(0, dim_h, 2)[: (dim_h // 2)].float() / dim_h
    dim_temporal_range = torch.arange(0, dim_t, 2)[: (dim_t // 2)].float() / dim_t

    h_theta = 10000.0 * rope_emb.h_ntk_factor
    w_theta = 10000.0 * rope_emb.w_ntk_factor
    t_theta = 10000.0 * rope_emb.t_ntk_factor

    h_freqs = 1.0 / (h_theta ** dim_spatial_range)
    w_freqs = 1.0 / (w_theta ** dim_spatial_range)
    t_freqs = 1.0 / (t_theta ** dim_temporal_range)

    freqs_h = torch.outer(seq[:H], h_freqs)       # [H, dim_h//2]
    freqs_w = torch.outer(seq[:W], w_freqs)       # [W, dim_h//2]
    freqs_t = torch.outer(seq[:T], t_freqs)       # [T, dim_t//2]

    # Expand to [T, H, W, dim//2] then flatten to [T*H*W, dim//2]
    # Using expand + reshape instead of einops repeat
    freqs_t_exp = freqs_t.unsqueeze(1).unsqueeze(2).expand(T, H, W, -1)
    freqs_h_exp = freqs_h.unsqueeze(0).unsqueeze(2).expand(T, H, W, -1)
    freqs_w_exp = freqs_w.unsqueeze(0).unsqueeze(1).expand(T, H, W, -1)

    freqs = torch.cat([freqs_t_exp, freqs_h_exp, freqs_w_exp], dim=-1)  # [T, H, W, D]
    return freqs.reshape(T * H * W, -1).float()


# ---------------------------------------------------------------------------
# Sinusoidal embedding (ONNX-traceable version)
# ---------------------------------------------------------------------------


def onnx_sinusoidal_embedding_1d(dim, position):
    """Sinusoidal embedding that avoids float64 for ONNX compatibility."""
    half = dim // 2
    position = position.float()
    sinusoid = torch.outer(position, torch.pow(10000.0, -torch.arange(half, device=position.device, dtype=torch.float32) / half))
    return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)


# ---------------------------------------------------------------------------
# ONNX-exportable wrapper module
# ---------------------------------------------------------------------------


class WanModelONNX(nn.Module):
    """
    Wrapper around WanModel that replaces all non-traceable ops with
    ONNX-compatible pure PyTorch equivalents.

    RoPE frequencies are pre-computed for fixed T, H, W and stored as a buffer.
    The forward pass uses only traceable operations.
    """

    def __init__(self, model: WanModel, T: int, H: int, W: int):
        super().__init__()
        self.model = model
        self.patch_size = model.patch_size
        self.in_dim = model.in_dim
        self.out_dim = model.out_dim
        self.dim = model.dim
        self.freq_dim = model.freq_dim
        self.num_heads = model.num_heads

        kt, kh, kw = self.patch_size
        self.T = T // kt
        self.H = H // kh
        self.W = W // kw
        self.kt = kt
        self.kh = kh
        self.kw = kw

        # Pre-compute RoPE frequencies as a buffer (not traced)
        freqs = precompute_rope_freqs(model.rope_position_embedding, self.T, self.H, self.W)
        self.register_buffer("rope_freqs", freqs)  # [L, head_dim//2]

    def forward(self, x_B_C_T_H_W, timesteps_B_T, crossattn_emb):
        B = x_B_C_T_H_W.shape[0]
        kt, kh, kw = self.kt, self.kh, self.kw
        T, H, W = self.T, self.H, self.W
        C_in = self.in_dim

        # --- Patchify (replaces einops rearrange) ---
        # x: [B, C, T*kt, H*kh, W*kw] -> [B, T, kt, H, kh, W, kw, C] -> [B, T*H*W, C*kt*kh*kw]
        x = x_B_C_T_H_W.reshape(B, C_in, T, kt, H, kh, W, kw)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7)     # [B, T, H, W, C, kt, kh, kw]
        x = x.reshape(B, T * H * W, C_in * kt * kh * kw)

        # --- Embeddings ---
        x = self.model.patch_embedding(x)

        # --- Time embedding ---
        t_B = timesteps_B_T[:, 0]
        e_B_D = self.model.time_embedding(onnx_sinusoidal_embedding_1d(self.freq_dim, t_B))
        e0_B_6_D = self.model.time_projection(e_B_D).reshape(B, 6, self.dim)

        # --- Text embedding ---
        context = self.model.text_embedding(crossattn_emb)

        # --- Transformer blocks ---
        freqs = self.rope_freqs  # [L, D//2]

        for block in self.model.blocks:
            x = self._block_forward(block, x, e0_B_6_D, freqs, context)

        # --- Head ---
        x = self._head_forward(self.model.head, x, e_B_D)

        # --- Unpatchify (replaces einops rearrange) ---
        # x: [B, T*H*W, kt*kh*kw*out_dim] -> [B, out_dim, T*kt, H*kh, W*kw]
        x = x.reshape(B, T, H, W, kt, kh, kw, self.out_dim)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)      # [B, out_dim, T, kt, H, kh, W, kw]
        x = x.reshape(B, self.out_dim, T * kt, H * kh, W * kw)

        return x

    def _block_forward(self, block, x, e, freqs, context):
        """Single transformer block forward (no autocast, no distributed)."""
        mod = (block.modulation + e.float()).chunk(6, dim=1)

        # Self-attention
        x_norm = block.norm1(x.float()).to(x.dtype)
        sa_in = (x_norm.float() * (1 + mod[1]) + mod[0]).to(x.dtype)
        y = self._self_attn_forward(block.self_attn, sa_in, freqs)
        x = x.float() + y.float() * mod[2]
        x = x.to(sa_in.dtype)

        # Cross-attention
        x_norm3 = block.norm3(x)
        x = x + self._cross_attn_forward(block.cross_attn, x_norm3, context)

        # FFN
        x_norm2 = block.norm2(x.float()).to(x.dtype)
        ffn_in = (x_norm2.float() * (1 + mod[4]) + mod[3]).to(x.dtype)
        y = block.ffn(ffn_in)
        x = x.float() + y.float() * mod[5]
        x = x.to(sa_in.dtype)

        return x

    def _self_attn_forward(self, attn, x, freqs):
        """Self-attention with pure PyTorch RoPE and SDPA."""
        B, S = x.shape[:2]
        n, d = attn.num_heads, attn.head_dim

        q = attn.norm_q(attn.q(x)).reshape(B, S, n, d)
        k = attn.norm_k(attn.k(x)).reshape(B, S, n, d)
        v = attn.v(x).reshape(B, S, n, d)

        q = onnx_rope_apply(q, freqs)
        k = onnx_rope_apply(k, freqs)

        out = onnx_sdpa(q, k, v)  # [B, S, H, D]
        out = out.reshape(B, S, n * d)
        return attn.o(out)

    def _cross_attn_forward(self, attn, x, context):
        """Cross-attention (T2V or I2V) with SDPA."""
        B = x.shape[0]
        n, d = attn.num_heads, attn.head_dim

        if isinstance(attn, WanI2VCrossAttention):
            img_len = context.shape[1] - T5_CONTEXT_TOKEN_NUMBER
            context_img = context[:, :img_len]
            context_txt = context[:, img_len:]

            q = attn.norm_q(attn.q(x)).reshape(B, -1, n, d)
            k = attn.norm_k(attn.k(context_txt)).reshape(B, -1, n, d)
            v = attn.v(context_txt).reshape(B, -1, n, d)
            k_img = attn.norm_k_img(attn.k_img(context_img)).reshape(B, -1, n, d)
            v_img = attn.v_img(context_img).reshape(B, -1, n, d)

            out = onnx_sdpa(q, k, v)
            out_img = onnx_sdpa(q, k_img, v_img)

            out = out.reshape(B, -1, n * d) + out_img.reshape(B, -1, n * d)
            return attn.o(out)
        else:
            q = attn.norm_q(attn.q(x)).reshape(B, -1, n, d)
            k = attn.norm_k(attn.k(context)).reshape(B, -1, n, d)
            v = attn.v(context).reshape(B, -1, n, d)

            out = onnx_sdpa(q, k, v)
            out = out.reshape(B, -1, n * d)
            return attn.o(out)

    def _head_forward(self, head, x, e):
        """Output head without autocast."""
        mod = (head.modulation + e.float().unsqueeze(1)).chunk(2, dim=1)
        x = head.head((head.norm(x.float()) * (1 + mod[1]) + mod[0]).to(x.dtype))
        return x


# ---------------------------------------------------------------------------
# Model creation
# ---------------------------------------------------------------------------


def select_model(model_name):
    """Create a vanilla WanModel (no SLA, no quantization)."""
    if model_name == "Wan2.1-1.3B":
        return WanModel(
            dim=1536, eps=1e-06, ffn_dim=8960, freq_dim=256, in_dim=16,
            model_type="t2v", num_heads=12, num_layers=30, out_dim=16, text_len=512,
        )
    elif model_name == "Wan2.1-14B":
        return WanModel(
            dim=5120, eps=1e-06, ffn_dim=13824, freq_dim=256, in_dim=16,
            model_type="t2v", num_heads=40, num_layers=40, out_dim=16, text_len=512,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}. Use Wan2.1-1.3B or Wan2.1-14B.")


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_onnx(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    # Latent spatial dimensions (after VAE 8x compression, 4x temporal)
    spatial_compression = 8
    temporal_compression = 4
    H_latent = args.height // spatial_compression
    W_latent = args.width // spatial_compression
    T_latent = (args.num_frames - 1) // temporal_compression + 1

    print(f"Creating model: {args.model}")
    with torch.device("meta"):
        base_model = select_model(args.model)

    print(f"Loading checkpoint: {args.dit_path}")
    state_dict = load_state_dict(args.dit_path)
    base_model.load_state_dict(state_dict, assign=True)
    del state_dict

    base_model = base_model.to(device=device, dtype=dtype).eval()

    print(f"Creating ONNX wrapper (T_latent={T_latent}, H_latent={H_latent}, W_latent={W_latent})")
    model = WanModelONNX(base_model, T=T_latent, H=H_latent, W=W_latent)
    model = model.to(device=device, dtype=dtype).eval()

    # Dummy inputs
    B = 1
    dummy_x = torch.randn(B, base_model.in_dim, T_latent, H_latent, W_latent, device=device, dtype=dtype)
    dummy_t = torch.randn(B, 1, device=device, dtype=dtype)
    dummy_emb = torch.randn(B, base_model.text_len, 4096, device=device, dtype=dtype)

    print(f"Dummy inputs: x={list(dummy_x.shape)}, t={list(dummy_t.shape)}, emb={list(dummy_emb.shape)}")

    # Verify forward pass works before export
    print("Running test forward pass...")
    with torch.no_grad():
        test_out = model(dummy_x, dummy_t, dummy_emb)
    print(f"Forward pass OK. Output shape: {list(test_out.shape)}")

    print(f"Exporting to ONNX: {args.output} (opset {args.opset})")
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy_x, dummy_t, dummy_emb),
            args.output,
            input_names=["x_B_C_T_H_W", "timesteps_B_T", "crossattn_emb"],
            output_names=["output"],
            dynamic_axes={
                "x_B_C_T_H_W": {0: "batch"},
                "timesteps_B_T": {0: "batch"},
                "crossattn_emb": {0: "batch"},
                "output": {0: "batch"},
            },
            opset_version=args.opset,
            do_constant_folding=True,
        )

    print(f"Exported: {args.output}")

    if args.verify:
        try:
            import onnx
            print("Verifying with onnx.checker...")
            onnx_model = onnx.load(args.output)
            onnx.checker.check_model(onnx_model)
            print("Verification passed.")
        except ImportError:
            print("Install 'onnx' to verify: pip install onnx")
        except Exception as e:
            print(f"Verification failed: {e}")

    if args.verify_runtime:
        try:
            import onnxruntime as ort
            import numpy as np

            print("Comparing PyTorch vs ONNX Runtime outputs...")
            sess = ort.InferenceSession(args.output)

            def to_np(t):
                return t.detach().cpu().float().numpy()

            ort_inputs = {
                "x_B_C_T_H_W": to_np(dummy_x),
                "timesteps_B_T": to_np(dummy_t),
                "crossattn_emb": to_np(dummy_emb),
            }
            ort_out = sess.run(None, ort_inputs)[0]
            pt_out = to_np(test_out)

            max_diff = np.abs(ort_out - pt_out).max()
            mean_diff = np.abs(ort_out - pt_out).mean()
            print(f"Max abs diff: {max_diff:.6f}, Mean abs diff: {mean_diff:.6f}")
            if max_diff < 0.1:
                print("Outputs match within tolerance.")
            else:
                print("WARNING: Large difference between PyTorch and ORT outputs.")
        except ImportError:
            print("Install 'onnxruntime' to compare: pip install onnxruntime-gpu")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Export TurboDiffusion WanModel to ONNX")
    parser.add_argument("--dit_path", type=str, required=True,
                        help="Path to the DiT model checkpoint (.pth)")
    parser.add_argument("--model", choices=["Wan2.1-1.3B", "Wan2.1-14B"], default="Wan2.1-1.3B",
                        help="Model variant to export")
    parser.add_argument("--output", type=str, default="wan_dit.onnx",
                        help="Output ONNX file path")
    parser.add_argument("--opset", type=int, default=18,
                        help="ONNX opset version (default: 18)")
    parser.add_argument("--num_frames", type=int, default=81,
                        help="Number of video frames (determines temporal latent size)")
    parser.add_argument("--height", type=int, default=480,
                        help="Video height in pixels (determines spatial latent size)")
    parser.add_argument("--width", type=int, default=832,
                        help="Video width in pixels (determines spatial latent size)")
    parser.add_argument("--verify", action="store_true",
                        help="Verify exported model with onnx.checker")
    parser.add_argument("--verify_runtime", action="store_true",
                        help="Compare PyTorch vs ONNX Runtime outputs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    export_onnx(args)
