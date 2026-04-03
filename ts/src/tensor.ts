/**
 * Lightweight tensor utilities for the sampling loop.
 * All heavy lifting (model inference) happens in ONNX Runtime.
 */

/** Create a Float32Array filled with normally-distributed random values (Box-Muller). */
export function randn(length: number, seed?: number): Float32Array {
  const out = new Float32Array(length);
  const rng = seed !== undefined ? mulberry32(seed) : Math.random;

  for (let i = 0; i < length; i += 2) {
    const u1 = rng();
    const u2 = rng();
    const r = Math.sqrt(-2 * Math.log(u1 || 1e-30));
    out[i] = r * Math.cos(2 * Math.PI * u2);
    if (i + 1 < length) {
      out[i + 1] = r * Math.sin(2 * Math.PI * u2);
    }
  }
  return out;
}

/** Simple seedable PRNG (mulberry32). */
function mulberry32(seed: number): () => number {
  let s = seed | 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Element-wise: out = (1 - tNext) * (x - tCur * vPred) + tNext * noise */
export function flowStep(
  x: Float32Array,
  vPred: Float32Array,
  noise: Float32Array,
  tCur: number,
  tNext: number,
): Float32Array {
  const out = new Float32Array(x.length);
  const oneMinusNext = 1 - tNext;
  for (let i = 0; i < x.length; i++) {
    out[i] = oneMinusNext * (x[i] - tCur * vPred[i]) + tNext * noise[i];
  }
  return out;
}

/** Compute the product of array elements. */
export function prod(arr: number[]): number {
  return arr.reduce((a, b) => a * b, 1);
}
