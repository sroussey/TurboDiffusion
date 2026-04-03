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

This script replaces non-ONNX-compatible operations (flash_attn rotary embeddings,
distributed attention wrappers, amp.autocast contexts) with pure PyTorch equivalents,
then exports the model via torch.onnx.export().

Usage:
    python export_onnx.py --model Wan2.1-1.3B --dit_path checkpoints/model.pth --output model.onnx

The exported model takes:
    - x_B_C_T_H_W: [B, C_in, T, H, W] input video tensor (float16)
    - timesteps_B_T: [B, 1] diffusion timesteps (float16)
    - crossattn_emb: [B, L, D] text embeddings (float16)

And returns:
    - output: [B, C_out, T, H, W] denoised video tensor (float16)
"""

import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from rcm.utils.model_utils import load_state_dict
from rcm.networks.wan2pt1 import (
    WanModel,
    WanSelfAttention,
    WanT2VCrossAttention,
    WanI2VCrossAttention,
    WanAttentionBlock,
    WanLayerNorm,
    Head,
    VideoRopePosition3DEmb,
    sinusoidal_embedding_1d,
)


# ---------------------------------------------------------------------------
# Pure PyTorch replacements for non-ONNX-compatible ops
# ---------------------------------------------------------------------------


def onnx_rope_apply(x, freqs):
    """
    Pure PyTorch interleaved rotary position embedding.
    Replaces flash_attn's apply_rotary_emb which is not ONNX-exportable.

    Args:
        x: [batch_size, seq_len, n_heads, head_dim]
        freqs: [seq_len, head_dim // 2]
    Returns:
        Tensor with same shape as x, with rotary embeddings applied.
    """
    batch_size, seq_len, n_heads, head_dim = x.shape

    freqs = freqs.view(seq_len, head_dim // 2)
    cos = torch.cos(freqs).to(x.dtype)  # [seq_len, head_dim//2]
    sin = torch.sin(freqs).to(x.dtype)

    # Interleaved layout: pairs are (x[..., 0], x[..., 1]), (x[..., 2], x[..., 3]), ...
    x = x.view(batch_size, seq_len, n_heads, head_dim // 2, 2)
    x0 = x[..., 0]  # [B, S, H, D//2]
    x1 = x[..., 1]  # [B, S, H, D//2]

    # Broadcast cos/sin: [S, D//2] -> [1, S, 1, D//2]
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)

    # Apply rotation
    o0 = x0 * cos - x1 * sin
    o1 = x0 * sin + x1 * cos

    # Interleave back
    out = torch.stack([o0, o1], dim=-1)  # [B, S, H, D//2, 2]
    return out.view(batch_size, seq_len, n_heads, head_dim)


def onnx_attention(q, k, v, dropout_p=0.0, softmax_scale=None, q_scale=None,
                   causal=False, deterministic=False):
    """
    ONNX-compatible attention using F.scaled_dot_product_attention directly.
    Replaces the original attention() which has runtime GPU capability checks
    and sdpa_kernel context managers that don't trace.
    """
    if q_scale is not None:
        q = q * q_scale

    # q, k, v: [B, S, H, D] -> [B, H, S, D]
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    out = F.scaled_dot_product_attention(
        q, k, v, is_causal=causal, dropout_p=dropout_p, scale=softmax_scale,
    )

    return out.transpose(1, 2).contiguous()


# ---------------------------------------------------------------------------
# Patched forward methods that remove amp.autocast and use ONNX-friendly ops
# ---------------------------------------------------------------------------


def _patched_self_attn_forward(self, x, seq_lens, freqs):
    """WanSelfAttention.forward without distributed ops."""
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

    q = self.norm_q(self.q(x)).view(b, s, n, d)
    k = self.norm_k(self.k(x)).view(b, s, n, d)
    v = self.v(x).view(b, s, n, d)

    q = onnx_rope_apply(q, freqs)
    k = onnx_rope_apply(k, freqs)

    x = onnx_attention(q, k, v)
    x = rearrange(x, "b s h d -> b s (h d)")
    x = self.o(x)
    return x


def _patched_t2v_cross_attn_forward(self, x, context, context_lens):
    """WanT2VCrossAttention.forward without distributed ops."""
    b, n, d = x.size(0), self.num_heads, self.head_dim

    q = self.norm_q(self.q(x)).view(b, -1, n, d)
    k = self.norm_k(self.k(context)).view(b, -1, n, d)
    v = self.v(context).view(b, -1, n, d)

    x = onnx_attention(q, k, v)
    x = rearrange(x, "b s h d -> b s (h d)")
    x = self.o(x)
    return x


def _patched_i2v_cross_attn_forward(self, x, context, context_lens):
    """WanI2VCrossAttention.forward without distributed ops."""
    from rcm.networks.wan2pt1 import T5_CONTEXT_TOKEN_NUMBER

    image_context_length = context.shape[1] - T5_CONTEXT_TOKEN_NUMBER
    context_img = context[:, :image_context_length]
    context = context[:, image_context_length:]
    b, n, d = x.size(0), self.num_heads, self.head_dim

    q = self.norm_q(self.q(x)).view(b, -1, n, d)
    k = self.norm_k(self.k(context)).view(b, -1, n, d)
    v = self.v(context).view(b, -1, n, d)
    k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
    v_img = self.v_img(context_img).view(b, -1, n, d)

    img_x = onnx_attention(q, k_img, v_img)
    x = onnx_attention(q, k, v)

    x = rearrange(x, "b s h d -> b s (h d)")
    img_x = rearrange(img_x, "b s h d -> b s (h d)")
    x = x + img_x
    x = self.o(x)
    return x


def _patched_block_forward(self, x, e, seq_lens, freqs, context, context_lens):
    """WanAttentionBlock.forward without amp.autocast."""
    e = (self.modulation + e.float()).chunk(6, dim=1)

    # self-attention
    y = self.self_attn((self.norm1(x).float() * (1 + e[1]) + e[0]).type_as(x), seq_lens, freqs)
    x = x + y * e[2].type_as(x)

    # cross-attention & ffn
    x = x + self.cross_attn(self.norm3(x), context, context_lens)
    y = self.ffn((self.norm2(x).float() * (1 + e[4]) + e[3]).type_as(x))
    x = x + y * e[5].type_as(x)
    return x


def _patched_head_forward(self, x, e):
    """Head.forward without amp.autocast."""
    e = (self.modulation + e.float().unsqueeze(1)).chunk(2, dim=1)
    x = self.head(self.norm(x).float() * (1 + e[1]) + e[0])
    return x


def _patched_wan_forward(
    self,
    x_B_C_T_H_W,
    timesteps_B_T,
    crossattn_emb,
    frame_cond_crossattn_emb_B_L_D=None,
    y_B_C_T_H_W=None,
):
    """
    WanModel.forward without distributed/context-parallel ops and without amp.autocast.
    """
    assert timesteps_B_T.shape[1] == 1
    t_B = timesteps_B_T[:, 0]

    if y_B_C_T_H_W is not None:
        x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, y_B_C_T_H_W], dim=1)

    kt, kh, kw = self.patch_size
    B, _, T_in, H_in, W_in = x_B_C_T_H_W.shape
    T, H, W = T_in // kt, H_in // kh, W_in // kw

    # patchify and flatten
    x_B_L_Din = rearrange(
        x_B_C_T_H_W,
        "b c (t kt) (h kh) (w kw) -> b (t h w) (c kt kh kw)",
        kt=kt, kh=kh, kw=kw,
    ).contiguous()

    # embeddings
    x_B_L_D = self.patch_embedding(x_B_L_Din)
    seq_lens = torch.tensor([x_B_L_D.size(1)] * B, dtype=torch.long, device=x_B_L_D.device)

    # time embeddings (no autocast)
    e_B_D = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t_B).float())
    e0_B_6_D = self.time_projection(e_B_D).unflatten(1, (6, self.dim))

    # context
    context_lens = None
    context_B_L_D = self.text_embedding(crossattn_emb)

    if frame_cond_crossattn_emb_B_L_D is not None:
        context_clip = self.img_emb(frame_cond_crossattn_emb_B_L_D)
        context_B_L_D = torch.cat([context_clip, context_B_L_D], dim=1)

    freqs = self.rope_position_embedding.generate_embeddings(
        torch.Size([B, T, H, W, self.dim])
    ).contiguous()

    kwargs = dict(
        e=e0_B_6_D,
        seq_lens=seq_lens,
        freqs=freqs,
        context=context_B_L_D,
        context_lens=context_lens,
    )

    for block in self.blocks:
        x_B_L_D = block(x_B_L_D, **kwargs)

    # head
    x_B_L_Dout = self.head(x_B_L_D, e_B_D)

    # unpatchify
    x_B_C_T_H_W = rearrange(
        x_B_L_Dout,
        "b (t h w) (kt kh kw d) -> b d (t kt) (h kh) (w kw)",
        kt=kt, kh=kh, kw=kw, t=T, h=H, w=W, d=self.out_dim,
    )
    return x_B_C_T_H_W


# ---------------------------------------------------------------------------
# RoPE patching: make generate_embeddings work without caching to cuda
# ---------------------------------------------------------------------------


def _patched_generate_embeddings(self, B_T_H_W_C, h_ntk_factor=None, w_ntk_factor=None, t_ntk_factor=None):
    """VideoRopePosition3DEmb.generate_embeddings that works on any device."""
    h_ntk_factor = h_ntk_factor if h_ntk_factor is not None else self.h_ntk_factor
    w_ntk_factor = w_ntk_factor if w_ntk_factor is not None else self.w_ntk_factor
    t_ntk_factor = t_ntk_factor if t_ntk_factor is not None else self.t_ntk_factor

    dim_h = self._dim_h
    dim_t = self._dim_t

    device = "cuda" if torch.cuda.is_available() else "cpu"

    seq = torch.arange(max(self.max_h, self.max_w, self.max_t), device=device).float()
    dim_spatial_range = torch.arange(0, dim_h, 2, device=device)[: (dim_h // 2)].float() / dim_h
    dim_temporal_range = torch.arange(0, dim_t, 2, device=device)[: (dim_t // 2)].float() / dim_t

    h_theta = 10000.0 * h_ntk_factor
    w_theta = 10000.0 * w_ntk_factor
    t_theta = 10000.0 * t_ntk_factor

    h_spatial_freqs = 1.0 / (h_theta ** dim_spatial_range)
    w_spatial_freqs = 1.0 / (w_theta ** dim_spatial_range)
    temporal_freqs = 1.0 / (t_theta ** dim_temporal_range)

    B, T, H, W, _ = B_T_H_W_C

    freqs_h = torch.outer(seq[:H], h_spatial_freqs)
    freqs_w = torch.outer(seq[:W], w_spatial_freqs)
    freqs_t = torch.outer(seq[:T], temporal_freqs)

    freqs_T_H_W_D = torch.cat(
        [
            repeat(freqs_t, "t d -> t h w d", h=H, w=W),
            repeat(freqs_h, "h d -> t h w d", t=T, w=W),
            repeat(freqs_w, "w d -> t h w d", t=T, h=H),
        ],
        dim=-1,
    )

    return rearrange(freqs_T_H_W_D, "t h w d -> (t h w) d").float()


# ---------------------------------------------------------------------------
# Monkey-patching and model setup
# ---------------------------------------------------------------------------


def patch_model_for_onnx(model):
    """
    Replace all non-ONNX-compatible operations in the model with pure PyTorch equivalents.
    """
    import types

    # Patch WanModel.forward
    model.forward = types.MethodType(_patched_wan_forward, model)

    # Patch RoPE embedding generation
    model.rope_position_embedding.generate_embeddings = types.MethodType(
        _patched_generate_embeddings, model.rope_position_embedding
    )

    # Patch all attention blocks
    for block in model.blocks:
        # Patch block forward (remove amp.autocast)
        block.forward = types.MethodType(_patched_block_forward, block)

        # Patch self-attention (remove distributed ops, use pure PyTorch RoPE)
        block.self_attn.forward = types.MethodType(_patched_self_attn_forward, block.self_attn)

        # Patch cross-attention
        if isinstance(block.cross_attn, WanI2VCrossAttention):
            block.cross_attn.forward = types.MethodType(_patched_i2v_cross_attn_forward, block.cross_attn)
        else:
            block.cross_attn.forward = types.MethodType(_patched_t2v_cross_attn_forward, block.cross_attn)

    # Patch head (remove amp.autocast)
    model.head.forward = types.MethodType(_patched_head_forward, model.head)

    return model


def select_model(model_name):
    """Create a WanModel with the given configuration (no SLA, no quantization)."""
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
        raise ValueError(f"Unsupported model for ONNX export: {model_name}. "
                         "Only T2V models (Wan2.1-1.3B, Wan2.1-14B) are supported.")


def export_onnx(args):
    """Export the WanModel to ONNX format."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # ONNX does not support bfloat16; use float16 on CUDA, float32 on CPU
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Creating model: {args.model}")
    with torch.device("meta"):
        model = select_model(args.model)

    print(f"Loading checkpoint: {args.dit_path}")
    state_dict = load_state_dict(args.dit_path)
    model.load_state_dict(state_dict, assign=True)
    del state_dict

    model = model.to(device=device, dtype=dtype).eval()

    print("Patching model for ONNX export...")
    model = patch_model_for_onnx(model)

    # Create dummy inputs for tracing
    # Use small spatial dims for export; dynamic axes allow variable sizes at runtime
    B = 1
    C_in = model.in_dim   # 16
    T = 5                 # temporal frames (after VAE compression)
    H = 30                # spatial height (after VAE compression)
    W = 52                # spatial width (after VAE compression)
    text_dim = 4096       # umT5 embedding dimension
    text_len = model.text_len  # 512

    dummy_x = torch.randn(B, C_in, T, H, W, device=device, dtype=dtype)
    dummy_t = torch.randn(B, 1, device=device, dtype=dtype)
    dummy_emb = torch.randn(B, text_len, text_dim, device=device, dtype=dtype)

    input_names = ["x_B_C_T_H_W", "timesteps_B_T", "crossattn_emb"]
    output_names = ["output"]

    dynamic_axes = {
        "x_B_C_T_H_W": {0: "batch", 2: "time", 3: "height", 4: "width"},
        "timesteps_B_T": {0: "batch"},
        "crossattn_emb": {0: "batch", 1: "text_len"},
        "output": {0: "batch", 2: "time", 3: "height", 4: "width"},
    }

    print(f"Exporting to ONNX: {args.output}")
    print(f"  Dummy input shapes: x={list(dummy_x.shape)}, t={list(dummy_t.shape)}, emb={list(dummy_emb.shape)}")

    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy_x, dummy_t, dummy_emb),
            args.output,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=args.opset,
            do_constant_folding=True,
        )

    print(f"ONNX model exported to: {args.output}")

    # Verify the exported model
    if args.verify:
        try:
            import onnx
            print("Verifying ONNX model...")
            onnx_model = onnx.load(args.output)
            onnx.checker.check_model(onnx_model)
            print("ONNX model verification passed.")
        except ImportError:
            print("Install 'onnx' package to verify the exported model: pip install onnx")
        except Exception as e:
            print(f"ONNX verification failed: {e}")


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
    parser.add_argument("--verify", action="store_true",
                        help="Verify the exported ONNX model with onnx.checker")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    export_onnx(args)
