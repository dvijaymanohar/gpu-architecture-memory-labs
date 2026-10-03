# Latency hiding and occupancy

Latency hiding is the GPU's ability to make useful progress by scheduling another eligible warp while a warp waits on memory or an instruction dependency.

Occupancy is the ratio of active warps to the architectural maximum. It can help latency hiding, but maximum occupancy is not itself the optimization target.

## Mechanisms
- SM warp schedulers select eligible warps.
- Long-latency memory and dependency stalls make some warps temporarily ineligible.
- Registers, shared memory, block size, and architectural limits constrain active blocks/warps.
- Too little parallel work can leave the GPU unable to hide latency.
- Raising occupancy can hurt if it causes spills, reduces per-thread resources, or changes the kernel unfavorably.

## Required experiments
1. Sweep block size and record runtime plus achieved occupancy.
2. Increase register pressure and inspect occupancy/spills.
3. Compare too little work with enough parallel work.
4. Produce one case where higher occupancy is not faster.
5. Use Nsight Compute stall-reason evidence to explain the result.

Always connect occupancy to the bottleneck mechanism rather than treating it as a score.
