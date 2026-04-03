"""
Export the umT5-XXL text encoder to ONNX format.

The encoder takes token IDs and attention mask, returns hidden states [B, 512, 4096].

Usage (from turbodiffusion/inference/):
    python export_t5_onnx.py \
        --text_encoder_path ../../checkpoints/models_t5_umt5-xxl-enc-bf16.pth \
        --output ../../onnx_model/wan_t5_encoder.onnx

Input:  input_ids [B, 512] int64, attention_mask [B, 512] int64
Output: hidden_states [B, 512, 4096] float32
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn

from rcm.utils.umt5 import umt5_xxl


class T5EncoderONNX(nn.Module):
    """ONNX-exportable wrapper for the umT5-XXL encoder."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)


def export_t5(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading umT5-XXL encoder: {args.text_encoder_path}")
    # Create model on meta device, then load weights
    with torch.device("meta"):
        model = umt5_xxl(encoder_only=True, dtype=torch.float32, device="meta")

    state_dict = torch.load(args.text_encoder_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict, assign=True)
    del state_dict

    model = model.to(device=device, dtype=torch.float32).eval()
    wrapper = T5EncoderONNX(model)

    seq_len = args.seq_len
    dummy_ids = torch.ones(1, seq_len, dtype=torch.long, device=device)
    dummy_mask = torch.ones(1, seq_len, dtype=torch.long, device=device)

    print(f"Test forward pass (seq_len={seq_len})...")
    with torch.no_grad():
        test_out = wrapper(dummy_ids, dummy_mask)
    print(f"Output shape: {list(test_out.shape)}")

    print(f"Exporting to: {args.output}")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_ids, dummy_mask),
            args.output,
            input_names=["input_ids", "attention_mask"],
            output_names=["hidden_states"],
            dynamic_axes={
                "input_ids": {0: "batch"},
                "attention_mask": {0: "batch"},
                "hidden_states": {0: "batch"},
            },
            opset_version=18,
            do_constant_folding=True,
        )
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export umT5-XXL encoder to ONNX")
    parser.add_argument("--text_encoder_path", type=str, required=True)
    parser.add_argument("--output", type=str, default="wan_t5_encoder.onnx")
    parser.add_argument("--seq_len", type=int, default=512)
    export_t5(parser.parse_args())
