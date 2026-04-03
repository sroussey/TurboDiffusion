/**
 * Rectified Consistency Model (RCM) sampler.
 *
 * Implements the TrigFlow -> RectifiedFlow timestep schedule and
 * the flow-matching update step used by TurboDiffusion for 1-4 step inference.
 */

export interface TimestepSchedule {
  /** Timestep values after TrigFlow->RectifiedFlow conversion. Length = numSteps + 1. */
  steps: number[];
}

/**
 * Build the RCM timestep schedule.
 *
 * @param numSteps - Number of sampling steps (1-4)
 * @param sigmaMax - Initial sigma (default 80 for T2V, 200 for I2V)
 */
export function buildSchedule(numSteps: number, sigmaMax: number = 80): TimestepSchedule {
  // Mid-timesteps for 2-4 step schedules (TrigFlow parameterization)
  const allMidT = [1.5, 1.4, 1.0];
  const midT = allMidT.slice(0, numSteps - 1);

  // Full schedule in TrigFlow space: [atan(sigma_max), ...mid_t, 0]
  const trigSteps = [Math.atan(sigmaMax), ...midT, 0];

  // Convert TrigFlow -> RectifiedFlow: t = sin(s) / (cos(s) + sin(s))
  const steps = trigSteps.map((s) => {
    const sinS = Math.sin(s);
    const cosS = Math.cos(s);
    return sinS / (cosS + sinS);
  });

  return { steps };
}
