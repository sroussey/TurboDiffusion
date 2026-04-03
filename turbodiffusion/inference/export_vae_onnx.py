"""
Export the Wan2.1 VAE decoder to ONNX format.

The VAE normally decodes one temporal frame at a time using causal convolution
caching. This wrapper processes all frames in a single pass (the CausalConv3d
already has correct causal padding, so single-pass produces identical output).

Usage (from turbodiffusion/inference/):
    python export_vae_onnx.py \
        --vae_path ../../checkpoints/Wan2.1_VAE.pth \
        --output ../../onnx_model/wan_vae_decoder.onnx

Input:  latent [B, 16, T_latent, H_latent, W_latent] float32
Output: video  [B, 3, T_pixel, H_pixel, W_pixel] float32
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn

from rcm.tokenizers.wan2pt1 import WanVAE_, _video_vae


class VAEDecoderONNX(nn.Module):
    """
    ONNX-exportable wrapper for the Wan2.1 VAE decoder.

    Processes all temporal frames in a single pass (no stateful caching).
    The scale (mean/std normalization) is baked in as buffers.
    """

    def __init__(self, vae: WanVAE_):
        super().__init__()
        self.conv2 = vae.conv2
        self.decoder = vae.decoder
        self.z_dim = vae.z_dim
        self.register_buffer("scale_mean", vae.mean.clone())
        self.register_buffer("scale_inv_std", (1.0 / vae.std).clone())

    def forward(self, z):
        # Un-normalize latents
        z = z * self.scale_inv_std.view(1, self.z_dim, 1, 1, 1) + self.scale_mean.view(1, self.z_dim, 1, 1, 1)
        # Project and decode all frames at once (feat_cache=None -> single-pass)
        x = self.conv2(z)
        # head layers
        x = self.decoder.conv1(x)
        for layer in self.decoder.middle:
            x = layer(x)
        for layer in self.decoder.upsamples:
            x = layer(x)
        for layer in self.decoder.head:
            x = layer(x)
        return x


def export_vae(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32  # VAE runs in float32 for quality

    print(f"Loading VAE: {args.vae_path}")
    vae_model = _video_vae(pretrained_path=args.vae_path, z_dim=16, device=device)
    vae_model = vae_model.to(dtype=dtype).eval()

    wrapper = VAEDecoderONNX(vae_model).to(device=device, dtype=dtype).eval()
    del vae_model

    T_latent = 1 + (args.num_frames - 1) // 4
    H_latent = args.height // 8
    W_latent = args.width // 8

    dummy_z = torch.randn(1, 16, T_latent, H_latent, W_latent, device=device, dtype=dtype)

    print(f"Test forward pass (latent shape: {list(dummy_z.shape)})...")
    with torch.no_grad():
        test_out = wrapper(dummy_z)
    print(f"Output shape: {list(test_out.shape)}")

    print(f"Exporting to: {args.output}")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_z,),
            args.output,
            input_names=["latent"],
            output_names=["video"],
            dynamic_axes={
                "latent": {0: "batch", 2: "T", 3: "H", 4: "W"},
                "video": {0: "batch"},
            },
            opset_version=18,
            do_constant_folding=True,
        )
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Wan2.1 VAE decoder to ONNX")
    parser.add_argument("--vae_path", type=str, required=True)
    parser.add_argument("--output", type=str, default="wan_vae_decoder.onnx")
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    export_vae(parser.parse_args())
