#include <cuda_runtime.h>
#include <iostream>

int main() {
    int count=0; cudaGetDeviceCount(&count);
    if(count==0){ std::cerr<<"No CUDA device\n"; return 77; }
    for(int d=0; d<count; ++d){
        cudaDeviceProp p{}; cudaGetDeviceProperties(&p,d);
        std::cout<<"device="<<d<<" name="<<p.name
                 <<" cc="<<p.major<<"."<<p.minor
                 <<" SMs="<<p.multiProcessorCount
                 <<" warp="<<p.warpSize
                 <<" global_mem_GB="<<(double)p.totalGlobalMem/1e9
                 <<" shared_per_block="<<p.sharedMemPerBlock
                 <<" regs_per_block="<<p.regsPerBlock<<"\n";
    }
}
