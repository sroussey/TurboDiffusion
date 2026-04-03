"""
ONNX-based inference for TurboDiffusion Wan2.1 text-to-video.

This script runs the full T2V pipeline using an ONNX-exported DiT backbone:
  1. T5 text encoder (PyTorch) encodes the prompt
  2. RCM sampling loop (1-4 steps) runs the ONNX DiT model
  3. VAE decoder (PyTorch) converts latents to video pixels
  4. Saves the output as .mp4

Usage (from turbodiffusion/inference/):
    python infer_onnx.py \
        --onnx_path ../../onnx_model/wan_dit.onnx \
        --text_encoder_path ../../checkpoints/models_t5_umt5-xxl-enc-bf16.pth \
        --vae_path ../../checkpoints/Wan2.1_VAE.pth \
        --prompt "A cat walking on a beach at sunset" \
        --save_path output/video.mp4

Prerequisites:
    pip install onnxruntime-gpu  # or onnxruntime for CPU-only
    # T5 encoder and VAE still require PyTorch + CUDA

If using the split ONNX files, first reassemble:
    cd onnx_model && bash reassemble.sh
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from tqdm import tqdm

try:
    import onnxruntime as ort
except ImportError:
    print("onnxruntime not found. Install with: pip install onnxruntime-gpu")
    sys.exit(1)

from rcm.utils.umt5 import clear_umt5_memory, get_umt5_embedding
from rcm.tokenizers.wan2pt1 import Wan2pt1VAEInterface


# Resolution presets (width, height)
VIDEO_RES = {
    "480p": {"1:1": (640, 640), "4:3": (640, 480), "3:4": (480, 640), "16:9": (832, 480), "9:16": (480, 832)},
    "720p": {"1:1": (960, 960), "4:3": (960, 720), "3:4": (720, 960), "16:9": (1280, 720), "9:16": (720, 1280)},
}


def save_video_torchvision(tensor, path, fps=16):
    """
    Save a [C, T, H, W] float tensor in [0, 1] range as an mp4.
    Uses torchvision if available, falls back to manual imageio/cv2.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    # [C, T, H, W] -> [T, H, W, C], uint8
    video = tensor.clamp(0, 1).permute(1, 2, 3, 0).mul(255).byte().cpu().numpy()

    try:
        import torchvision.io as tio
        # torchvision expects [T, H, W, C] uint8
        tio.write_video(path, torch.from_numpy(video), fps=fps)
        return
    except (ImportError, Exception):
        pass

    try:
        import imageio
        writer = imageio.get_writer(path, fps=fps)
        for frame in video:
            writer.append_data(frame)
        writer.close()
        return
    except ImportError:
        pass

    try:
        import cv2
        h, w = video.shape[1], video.shape[2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(path, fourcc, fps, (w, h))
        for frame in video:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()
        return
    except ImportError:
        pass

    raise RuntimeError(
        "No video writer found. Install one of: torchvision, imageio[ffmpeg], opencv-python"
    )


def create_ort_session(onnx_path, device="cuda"):
    """Create an ONNX Runtime inference session."""
    providers = []
    if device == "cuda":
        providers.append(("CUDAExecutionProvider", {"device_id": 0}))
    providers.append("CPUExecutionProvider")

    sess = ort.InferenceSession(onnx_path, providers=providers)
    active = sess.get_providers()
    print(f"ONNX Runtime providers: {active}")
    return sess


def ort_forward(sess, x, timesteps, crossattn_emb):
    """
    Run a single DiT forward pass through ONNX Runtime.

    Args:
        sess: ORT InferenceSession
        x: [B, 16, T, H, W] numpy float32
        timesteps: [B, 1] numpy float32
        crossattn_emb: [B, 512, 4096] numpy float32

    Returns:
        output: [B, 16, T, H, W] numpy float32
    """
    return sess.run(None, {
        "x_B_C_T_H_W": x,
        "timesteps_B_T": timesteps,
        "crossattn_emb": crossattn_emb,
    })[0]


def run_pipeline(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Resolution ---
    w, h = VIDEO_RES[args.resolution][args.aspect_ratio]
    T_latent = 1 + (args.num_frames - 1) // 4
    H_latent = h // 8
    W_latent = w // 8
    print(f"Output: {w}x{h}, {args.num_frames} frames")
    print(f"Latent: [B, 16, {T_latent}, {H_latent}, {W_latent}]")

    # --- 1. Text encoding (PyTorch) ---
    print(f"Encoding prompt: {args.prompt}")
    with torch.no_grad():
        text_emb = get_umt5_embedding(
            checkpoint_path=args.text_encoder_path,
            prompts=args.prompt,
        ).to(device=device)
    clear_umt5_memory()

    # Convert to numpy float32 for ORT, expand for num_samples
    text_emb_np = text_emb.float().cpu().numpy()
    if args.num_samples > 1:
        text_emb_np = np.repeat(text_emb_np, args.num_samples, axis=0)

    # --- 2. Load ONNX DiT model ---
    print(f"Loading ONNX model: {args.onnx_path}")
    sess = create_ort_session(args.onnx_path, device=device)

    # --- 3. RCM sampling loop ---
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    # Initial noise
    init_noise = torch.randn(
        args.num_samples, 16, T_latent, H_latent, W_latent,
        dtype=torch.float32, device=device, generator=generator,
    )

    # Timestep schedule (TrigFlow -> RectifiedFlow)
    mid_t = [1.5, 1.4, 1.0][: args.num_steps - 1]
    t_steps = torch.tensor(
        [math.atan(args.sigma_max), *mid_t, 0],
        dtype=torch.float64, device=device,
    )
    t_steps = torch.sin(t_steps) / (torch.cos(t_steps) + torch.sin(t_steps))

    x = init_noise.to(torch.float64) * t_steps[0]
    B = x.shape[0]
    ones = np.ones((B, 1), dtype=np.float32)

    print(f"Sampling {args.num_steps} steps (sigma_max={args.sigma_max})...")
    for i, (t_cur, t_next) in enumerate(tqdm(
        list(zip(t_steps[:-1], t_steps[1:])),
        desc="Sampling", total=len(t_steps) - 1,
    )):
        # Prepare inputs as numpy float32
        x_np = x.float().cpu().numpy()
        t_np = (ones * float(t_cur) * 1000).astype(np.float32)

        # ONNX Runtime forward pass
        v_pred_np = ort_forward(sess, x_np, t_np, text_emb_np)
        v_pred = torch.from_numpy(v_pred_np).to(dtype=torch.float64, device=device)

        # Flow matching update with stochastic noise injection
        x = (1 - t_next) * (x - t_cur * v_pred) + t_next * torch.randn(
            *x.shape, dtype=torch.float32, device=device, generator=generator,
        )

    samples = x.float()

    # --- 4. VAE decode (PyTorch) ---
    print("Decoding latents with VAE...")
    tokenizer = Wan2pt1VAEInterface(vae_pth=args.vae_path)
    with torch.no_grad():
        video = tokenizer.decode(samples)  # [B, 3, T, H, W] in [-1, 1]

    # --- 5. Save video ---
    video = video.float().cpu()
    video = (1.0 + video.clamp(-1, 1)) / 2.0  # [-1,1] -> [0,1]

    # Take first sample: [3, T, H, W]
    output = video[0]
    save_video_torchvision(output, args.save_path, fps=16)
    print(f"Saved: {args.save_path}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="ONNX-based TurboDiffusion T2V inference")
    parser.add_argument("--onnx_path", type=str, required=True,
                        help="Path to the ONNX DiT model (wan_dit.onnx)")
    parser.add_argument("--text_encoder_path", type=str, default="checkpoints/models_t5_umt5-xxl-enc-bf16.pth",
                        help="Path to the umT5 text encoder checkpoint")
    parser.add_argument("--vae_path", type=str, default="checkpoints/Wan2.1_VAE.pth",
                        help="Path to the Wan2.1 VAE checkpoint")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Text prompt for video generation")
    parser.add_argument("--save_path", type=str, default="output/generated_video.mp4",
                        help="Output video path (.mp4)")
    parser.add_argument("--num_frames", type=int, default=81,
                        help="Number of video frames to generate")
    parser.add_argument("--resolution", choices=["480p", "720p"], default="480p",
                        help="Output resolution preset")
    parser.add_argument("--aspect_ratio", default="16:9",
                        help="Aspect ratio (e.g. 16:9, 9:16, 4:3, 1:1)")
    parser.add_argument("--num_steps", type=int, choices=[1, 2, 3, 4], default=4,
                        help="Number of RCM sampling steps (1-4)")
    parser.add_argument("--sigma_max", type=float, default=80,
                        help="Initial sigma for RCM noise schedule")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of videos to generate")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for reproducibility")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    run_pipeline(args)
