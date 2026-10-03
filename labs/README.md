# Learn-by-doing labs

```bash
nvcc -O2 examples/thread_mapping.cu -o thread_mapping
./thread_mapping

nvcc -O3 examples/shared_memory_reuse.cu -o shared_memory_reuse
./shared_memory_reuse

nvcc -O3 examples/bank_conflicts.cu -o bank_conflicts
ncu ./bank_conflicts

nvcc -O3 examples/occupancy_sweep.cu -o occupancy_sweep
./occupancy_sweep
```

Before profiling, write down what you expect to happen and why. Then use counters to test the mechanism.
