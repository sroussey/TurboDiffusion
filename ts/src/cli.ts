#!/usr/bin/env node
/**
 * CLI for TurboDiffusion ONNX inference.
 *
 * With T5 model:
 *   npx tsx src/cli.ts \
 *     --dit ../onnx_model/wan_dit.onnx \
 *     --vae ../onnx_model/wan_vae_decoder.onnx \
 *     --t5  ../onnx_model/wan_t5_encoder.onnx \
 *     --prompt "A cat walking on a beach at sunset"
 *
 * With pre-computed embedding (no T5 needed):
 *   npx tsx src/cli.ts \
 *     --dit ../onnx_model/wan_dit.onnx \
 *     --vae ../onnx_model/wan_vae_decoder.onnx \
 *     --embedding embedding.bin \
 *     --prompt unused
 */

import { readFileSync, writeFileSync } from "node:fs";
import { parseArgs } from "node:util";
import { TurboDiffusionPipeline } from "./pipeline.js";

const { values } = parseArgs({
  options: {
    dit: { type: "string" },
    vae: { type: "string" },
    t5: { type: "string" },
    "tokenizer-model": { type: "string" },
    embedding: { type: "string" },
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

if (!values.dit || !values.vae || (!values.prompt && !values.embedding)) {
  console.error("Required: --dit <path> --vae <path> --prompt <text>");
  console.error("  or:     --dit <path> --vae <path> --embedding <path>");
  console.error("");
  console.error("Options:");
  console.error("  --t5 <path>              T5 ONNX model (needed if using --prompt)");
  console.error("  --tokenizer-model <id>   HuggingFace tokenizer (default: google/umt5-xxl)");
  console.error("  --embedding <path>       Pre-computed .bin file [1,512,4096] float32");
  console.error("  --output <path>          Output file (default: output.raw)");
  console.error("  --steps 1-4              RCM sampling steps (default: 4)");
  console.error("  --seed <int>             Random seed (default: 0)");
  console.error("  --device cpu|cuda        Execution provider (default: cpu)");
  process.exit(1);
}

async function main() {
  const pipeline = new TurboDiffusionPipeline({
    ditPath: values.dit!,
    vaePath: values.vae!,
    t5Path: values.t5,
    tokenizerModel: values["tokenizer-model"],
    device: values.device as "cpu" | "cuda",
  });

  await pipeline.load();

  // Load pre-computed embedding if provided
  let precomputedEmbedding: Float32Array | undefined;
  if (values.embedding) {
    const buf = readFileSync(values.embedding);
    precomputedEmbedding = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
    console.log(`Loaded embedding: ${values.embedding} (${precomputedEmbedding.length} floats)`);
  }

  const result = await pipeline.generate({
    prompt: values.embedding ? undefined : values.prompt,
    precomputedEmbedding,
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

  const outPath = values.output!;
  writeFileSync(outPath, Buffer.from(result.video.buffer));
  console.log(`Saved: ${outPath}`);

  if (outPath.endsWith(".raw")) {
    console.log(`\nConvert to mp4:`);
    console.log(`  python -c "
import numpy as np, imageio
d = np.fromfile('${outPath}', dtype=np.float32).reshape(3, ${shape.frames}, ${shape.height}, ${shape.width})
d = (d.transpose(1, 2, 3, 0) * 255).clip(0, 255).astype(np.uint8)
writer = imageio.get_writer('video.mp4', fps=16)
for f in d: writer.append_data(f)
writer.close()
"`);
  }

  await pipeline.dispose();
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
