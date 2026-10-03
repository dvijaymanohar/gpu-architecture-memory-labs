# Profiling workflow

```bash
ncu --set full -o profiles/stride ./build/stride_benchmark
```

Inspect achieved memory throughput, sectors/requests, occupancy, eligible warps, register use, and stall reasons. Form one hypothesis before changing code.
