"""
Pre-compute T5 text embeddings and save as a .bin file for use in TypeScript.

Usage (from turbodiffusion/inference/):
    python precompute_embeddings.py \
        --text_encoder_path ../../checkpoints/models_t5_umt5-xxl-enc-bf16.pth \
        --prompt "A cat walking on a beach at sunset" \
        --output embedding.bin

The output is a raw float32 file of shape [1, 512, 4096] (8MB).
Load in TypeScript:
    const emb = new Float32Array(fs.readFileSync("embedding.bin").buffer);
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from rcm.utils.umt5 import get_umt5_embedding, clear_umt5_memory


def main():
    parser = argparse.ArgumentParser(description="Pre-compute T5 embeddings")
    parser.add_argument("--text_encoder_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--output", type=str, default="embedding.bin")
    args = parser.parse_args()

    print(f"Encoding: {args.prompt}")
    with torch.no_grad():
        emb = get_umt5_embedding(checkpoint_path=args.text_encoder_path, prompts=args.prompt)
    clear_umt5_memory()

    arr = emb.float().cpu().numpy()  # [1, 512, 4096]
    print(f"Shape: {arr.shape}, dtype: {arr.dtype}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    arr.tofile(args.output)
    print(f"Saved: {args.output} ({os.path.getsize(args.output) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
