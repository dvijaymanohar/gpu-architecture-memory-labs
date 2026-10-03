# gpu-architecture-memory-labs

Learn GPU execution and memory hierarchy by measuring SM/warp behavior, coalescing, shared memory, occupancy, latency hiding, precision, and Tensor Core eligibility.

This repository follows a **learn-by-example** progression:

> concept → runnable example → correctness → measurement → profiling → diagnosis → optimization → validation → project

## Basics coverage
GPU architecture; SMs; warps; SIMT; registers/shared/L1/L2/global memory; latency hiding; occupancy; coalescing; precision; Tensor Cores.

## Learning order
1. GPU properties
2. thread/warp/block mapping
3. stride benchmark
4. coalesced vs strided loads
5. shared-memory tiling
6. bank conflicts
7. cache reuse
8. FP32 vs FP16/BF16
9. Tensor Core matmul

## Repository layout
- `fundamentals/` — concise mechanism notes and tiny demonstrations
- `examples/` — runnable examples in learning order
- `tests/` — deterministic and randomized correctness checks
- `benchmarks/` — repeatable measurement harnesses
- `profiling/` — profiler commands and evidence instructions
- `optimizations/` — baseline → hypothesis → change → re-measure studies
- `exercises/` — beginner through challenge tasks
- `mini-projects/` — integrated practice
- `advanced-projects/` — portfolio-grade work
- `docs/` — deeper explanations and decision records
- `scripts/` — setup/environment helpers
- `references/` — primary-source references

## Working rules
1. Establish correctness before performance work.
2. Define the measurement boundary.
3. Warm up before steady-state measurements.
4. Repeat measurements and report median plus spread.
5. Profile before optimizing.
6. Change one major variable at a time.
7. Re-run correctness checks after every optimization.
8. Never commit invented benchmark numbers; record actual environment metadata.

## Environment
```bash
bash scripts/check_environment.sh
```

GPU examples require compatible NVIDIA hardware/software. Hardware-dependent work is explicitly marked rather than simulated.

## Completion standard
A topic is complete only when you can explain the mechanism, run/build the example, validate correctness, measure it correctly, interpret relevant profiler evidence, and explain the trade-offs.
