#!/usr/bin/env node
/**
 * CLI for TurboDiffusion ONNX inference.
 *
 * Usage:
 *   npx tsx src/cli.ts \
 *     --dit ../onnx_model/wan_dit.onnx \
 *     --vae ../onnx_model/wan_vae_decoder.onnx \
 *     --t5  ../onnx_model/wan_t5_encoder.onnx \
 *     --prompt "A cat walking on a beach at sunset" \
 *     --output video.mp4
 */

import { writeFileSync } from "node:fs";
import { parseArgs } from "node:util";
import { TurboDiffusionPipeline } from "./pipeline.js";

const { values } = parseArgs({
  options: {
    dit: { type: "string" },
    vae: { type: "string" },
    t5: { type: "string" },
    prompt: { type: "string" },
    output: { type: "string", default: "output.raw" },
    steps: { type: "string", default: "4" },
    seed: { type: "string", default: "0" },
    frames: { type: "string", default: "81" },
    width: { type: "string", default: "832" },
    height: { type: "string", default: "480" },
    device: { type: "string", default: "cpu" },
  },
});

if (!values.dit || !values.vae || !values.prompt) {
  console.error("Required: --dit <path> --vae <path> --prompt <text>");
  console.error("Optional: --t5 <path> --output <path> --steps 4 --seed 0 --device cpu|cuda");
  process.exit(1);
}

async function main() {
  const pipeline = new TurboDiffusionPipeline({
    ditPath: values.dit!,
    vaePath: values.vae!,
    t5Path: values.t5,
    device: values.device as "cpu" | "cuda",
  });

  await pipeline.load();

  const result = await pipeline.generate({
    prompt: values.prompt,
    numSteps: parseInt(values.steps!, 10),
    seed: parseInt(values.seed!, 10),
    numFrames: parseInt(values.frames!, 10),
    width: parseInt(values.width!, 10),
    height: parseInt(values.height!, 10),
    onProgress: (step, total) => {
      console.log(`Progress: ${step}/${total}`);
    },
  });

  const { shape } = result;
  console.log(`Video: ${shape.frames} frames, ${shape.width}x${shape.height}`);

  // Save raw float32 tensor (use ffmpeg or a script to convert to mp4)
  const outPath = values.output!;
  if (outPath.endsWith(".raw")) {
    writeFileSync(outPath, Buffer.from(result.video.buffer));
    console.log(`Saved raw tensor: ${outPath}`);
    console.log(`Convert to mp4 with:`);
    console.log(`  python -c "
import numpy as np, imageio
d = np.fromfile('${outPath}', dtype=np.float32).reshape(3, ${shape.frames}, ${shape.height}, ${shape.width})
d = (d.transpose(1, 2, 3, 0) * 255).clip(0, 255).astype(np.uint8)
writer = imageio.get_writer('video.mp4', fps=16)
for f in d: writer.append_data(f)
writer.close()
"`);
  } else {
    // For mp4 output, use the raw -> mp4 conversion inline
    writeFileSync(outPath, Buffer.from(result.video.buffer));
    console.log(`Saved: ${outPath} (raw float32, needs conversion to mp4)`);
  }

  await pipeline.dispose();
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
