# TODO

- Greedy (deterministic) keydoor DQN policy still stalls on 6/10 trials in
  `results/run_20261022_163907` (spins in place via `left`/`right` until
  truncation) even though action masking fixed the earlier illegal-action
  loop and training-time last-100-episode return looks solved (7/10 at
  2.000). Root cause: `BinarySuccessRewardWrapper` gives 0 reward on every
  non-terminal step, so with `gamma=0.995` there's little gradient pushing
  the network to prefer progress over stalling; training metrics look fine
  only because `exploration_final_eps=0.05` randomly kicks stuck episodes
  loose during training. Fix: add a small per-step penalty (e.g. `-0.01`)
  to give stalling an explicit cost. Consider pairing with Double DQN
  (SB3's plain `DQN` doesn't use it, which is known to cause this kind of
  greedy-optimum overestimation on rarely-visited loop states). Then rerun
  the test suite and the delay-0 GPU sweep to confirm the greedy policy
  actually solves it, not just the exploration-noised training return.
