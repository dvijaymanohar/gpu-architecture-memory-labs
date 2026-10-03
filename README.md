# GPU Architecture & Memory Labs

Measure how GPU execution hierarchy and memory hierarchy shape performance.

## Sequence
1. Query device properties.
2. Inspect thread/block/warp mapping.
3. Measure global-memory stride.
4. Compare coalesced vs strided access.
5. Reuse data in shared memory.
6. Create/avoid bank conflicts.
7. Explore cache reuse.
8. Compare FP32, FP16, and BF16 footprint/throughput.
9. Compare Tensor Core-eligible matrix multiplication.
10. Relate occupancy to latency hiding without treating occupancy as the goal.

## Build
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/device_properties
./build/stride_benchmark 16777216
```

Use Nsight Compute only after establishing a correct baseline. See `docs/latency-hiding-and-occupancy.md`.
