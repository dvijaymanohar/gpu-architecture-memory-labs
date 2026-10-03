# Learning path

The repository specification is implemented as a sequence of controlled experiments.

## Progression

- [ ] 01 — GPU properties
- [ ] 02 — thread/warp/block mapping
- [ ] 03 — stride benchmark
- [ ] 04 — coalesced vs strided loads
- [ ] 05 — shared-memory tiling
- [ ] 06 — bank conflicts
- [ ] 07 — cache reuse
- [ ] 08 — FP32 vs FP16/BF16
- [ ] 09 — Tensor Core matmul

## Evidence template

For every performance experiment, record:

- objective
- hypothesis
- workload and input shape
- independent variable
- controlled variables
- hardware/software environment
- correctness criterion
- timing methodology
- median and spread
- profiler evidence
- interpretation
- trade-off
- next experiment

Do not optimize a workload until the baseline is correct and the bottleneck hypothesis is supported by evidence.
