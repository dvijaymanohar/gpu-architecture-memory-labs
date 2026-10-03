// Build: nvcc -O2 examples/thread_mapping.cu -o thread_mapping
#include <cuda_runtime.h>
#include <cstdio>
__global__ void show_mapping(){
    int g=blockIdx.x*blockDim.x+threadIdx.x;
    int warp=threadIdx.x/warpSize, lane=threadIdx.x%warpSize;
    printf("block=%d thread=%d global=%d warp_in_block=%d lane=%d\n",blockIdx.x,threadIdx.x,g,warp,lane);
}
int main(){show_mapping<<<2,40>>>();cudaDeviceSynchronize();}
