/**
 * TurboDiffusion text-to-video pipeline using ONNX Runtime.
 *
 * Three ONNX models:
 *   1. T5 encoder   – text prompt -> [B, 512, 4096] embeddings
 *   2. DiT backbone  – denoising network (runs 1-4 times in RCM loop)
 *   3. VAE decoder   – latents [B, 16, T, H, W] -> video [B, 3, T', H', W']
 *
 * All models run through onnxruntime-node.
 */

import * as ort from "onnxruntime-node";
import { AutoTokenizer } from "@huggingface/transformers";
import { randn, flowStep, prod } from "./tensor.js";
import { buildSchedule } from "./sampler.js";

export interface PipelineConfig {
  /** Path to the DiT ONNX model (wan_dit.onnx). */
  ditPath: string;
  /** Path to the VAE decoder ONNX model (wan_vae_decoder.onnx). */
  vaePath: string;
  /**
   * Path to the T5 encoder ONNX model (wan_t5_encoder.onnx).
   * If omitted, you must pass pre-computed embeddings to generate().
   */
  t5Path?: string;
  /** ONNX Runtime execution provider. Default: 'cpu'. Set 'cuda' for GPU. */
  device?: "cpu" | "cuda";
}

export interface GenerateOptions {
  /** Text prompt. Required unless precomputedEmbedding is provided. */
  prompt?: string;
  /** Pre-computed T5 embedding [1, 512, 4096] as Float32Array. Skips T5 inference. */
  precomputedEmbedding?: Float32Array;
  /** Number of RCM sampling steps (1-4). Default: 4. */
  numSteps?: number;
  /** Initial sigma for noise schedule. Default: 80. */
  sigmaMax?: number;
  /** Number of video frames. Default: 81. */
  numFrames?: number;
  /** Video width in pixels. Default: 832. */
  width?: number;
  /** Video height in pixels. Default: 480. */
  height?: number;
  /** Random seed. Default: 0. */
  seed?: number;
  /** Progress callback, called after each sampling step. */
  onProgress?: (step: number, totalSteps: number) => void;
}

export interface GenerateResult {
  /** Raw video tensor [3, T_pixel, H_pixel, W_pixel] in [0, 1] range, row-major Float32Array. */
  video: Float32Array;
  /** Video dimensions. */
  shape: { channels: 3; frames: number; height: number; width: number };
}

export class TurboDiffusionPipeline {
  private ditSession: ort.InferenceSession | null = null;
  private vaeSession: ort.InferenceSession | null = null;
  private t5Session: ort.InferenceSession | null = null;
  private tokenizer: Awaited<ReturnType<typeof AutoTokenizer.from_pretrained>> | null = null;
  private config: PipelineConfig;

  constructor(config: PipelineConfig) {
    this.config = config;
  }

  /** Load all ONNX models. Call once before generate(). */
  async load(): Promise<void> {
    const opts: ort.InferenceSession.SessionOptions = {};
    if (this.config.device === "cuda") {
      opts.executionProviders = [{ name: "cuda", deviceId: 0 } as ort.InferenceSession.ExecutionProviderConfig];
    }

    console.log("Loading DiT model...");
    this.ditSession = await ort.InferenceSession.create(this.config.ditPath, opts);

    console.log("Loading VAE decoder...");
    this.vaeSession = await ort.InferenceSession.create(this.config.vaePath, opts);

    if (this.config.t5Path) {
      console.log("Loading T5 encoder...");
      this.t5Session = await ort.InferenceSession.create(this.config.t5Path, opts);
      console.log("Loading tokenizer (google/umt5-xxl)...");
      this.tokenizer = await AutoTokenizer.from_pretrained("google/umt5-xxl");
    }

    console.log("Pipeline ready.");
  }

  /** Encode a text prompt to [1, 512, 4096] embedding using the T5 encoder. */
  async encodeText(prompt: string): Promise<Float32Array> {
    if (!this.t5Session || !this.tokenizer) {
      throw new Error("T5 model not loaded. Provide t5Path in config or use precomputedEmbedding.");
    }

    const encoded = await this.tokenizer(prompt, {
      padding: "max_length",
      max_length: 512,
      truncation: true,
      return_tensors: "np",
    });

    const inputIds = new BigInt64Array(512);
    const attentionMask = new BigInt64Array(512);
    const ids = encoded.input_ids.data as number[];
    const mask = encoded.attention_mask.data as number[];
    for (let i = 0; i < 512; i++) {
      inputIds[i] = BigInt(ids[i] ?? 0);
      attentionMask[i] = BigInt(mask[i] ?? 0);
    }

    const result = await this.t5Session.run({
      input_ids: new ort.Tensor("int64", inputIds, [1, 512]),
      attention_mask: new ort.Tensor("int64", attentionMask, [1, 512]),
    });

    return result.hidden_states.data as Float32Array;
  }

  /** Run the full text-to-video pipeline. */
  async generate(options: GenerateOptions = {}): Promise<GenerateResult> {
    const {
      numSteps = 4,
      sigmaMax = 80,
      numFrames = 81,
      width = 832,
      height = 480,
      seed = 0,
      onProgress,
    } = options;

    if (!this.ditSession || !this.vaeSession) {
      throw new Error("Pipeline not loaded. Call load() first.");
    }

    // --- 1. Get text embeddings ---
    let embData: Float32Array;
    if (options.precomputedEmbedding) {
      embData = options.precomputedEmbedding;
    } else if (options.prompt) {
      console.log("Encoding prompt...");
      embData = await this.encodeText(options.prompt);
    } else {
      throw new Error("Either prompt or precomputedEmbedding is required.");
    }

    // --- 2. RCM sampling ---
    const T = 1 + Math.floor((numFrames - 1) / 4);
    const H = Math.floor(height / 8);
    const W = Math.floor(width / 8);
    const latentShape = [1, 16, T, H, W];
    const latentSize = prod(latentShape);

    console.log(`Latent shape: [1, 16, ${T}, ${H}, ${W}]`);

    const schedule = buildSchedule(numSteps, sigmaMax);
    const steps = schedule.steps;

    // Initialize x = noise * t_steps[0]
    let x = randn(latentSize, seed);
    const t0 = steps[0];
    for (let i = 0; i < x.length; i++) x[i] *= t0;

    const totalSteps = steps.length - 1;
    const ones = new Float32Array([1.0]);

    console.log(`Sampling ${totalSteps} steps...`);
    for (let i = 0; i < totalSteps; i++) {
      const tCur = steps[i];
      const tNext = steps[i + 1];

      // Timestep input: [B, 1] = tCur * 1000
      const timestep = new Float32Array([tCur * 1000]);

      // Run DiT forward pass
      const ditResult = await this.ditSession.run({
        x_B_C_T_H_W: new ort.Tensor("float32", x, latentShape),
        timesteps_B_T: new ort.Tensor("float32", timestep, [1, 1]),
        crossattn_emb: new ort.Tensor("float32", embData, [1, 512, 4096]),
      });

      const vPred = ditResult.output.data as Float32Array;

      // Flow matching update with stochastic noise
      const noise = randn(latentSize, seed + i + 1);
      x = flowStep(x, vPred, noise, tCur, tNext);

      onProgress?.(i + 1, totalSteps);
      console.log(`  Step ${i + 1}/${totalSteps} done`);
    }

    // --- 3. VAE decode ---
    console.log("Decoding with VAE...");
    const vaeResult = await this.vaeSession.run({
      latent: new ort.Tensor("float32", x, latentShape),
    });

    const videoData = vaeResult.video.data as Float32Array;
    const videoShape = vaeResult.video.dims as number[];
    // videoShape = [1, 3, T_pixel, H_pixel, W_pixel]

    const T_pixel = videoShape[2];
    const H_pixel = videoShape[3];
    const W_pixel = videoShape[4];

    // Remove batch dim -> [3, T, H, W] and clamp to [0, 1]: (x + 1) / 2
    const pixelCount = 3 * T_pixel * H_pixel * W_pixel;
    const video = new Float32Array(pixelCount);
    for (let i = 0; i < pixelCount; i++) {
      video[i] = Math.max(0, Math.min(1, (videoData[i] + 1) / 2));
    }

    return {
      video,
      shape: { channels: 3, frames: T_pixel, height: H_pixel, width: W_pixel },
    };
  }

  /** Release ONNX sessions. */
  async dispose(): Promise<void> {
    await this.ditSession?.release();
    await this.vaeSession?.release();
    await this.t5Session?.release();
    this.ditSession = null;
    this.vaeSession = null;
    this.t5Session = null;
  }
}
