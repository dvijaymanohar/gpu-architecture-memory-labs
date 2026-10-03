// Build: nvcc -O3 examples/occupancy_sweep.cu -o occupancy_sweep
#include <cuda_runtime.h>
#include <iostream>
__global__ void chain(float*x,int n,int iters){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    float v=x[i];
    for(int k=0;k<iters;++k)v=v*1.000001f+0.000001f;
    x[i]=v;
}
int main(){
    int n=1<<24,iters=128;float*x;cudaMalloc(&x,(size_t)n*4);cudaMemset(x,0,(size_t)n*4);
    for(int th: {64,128,256,512,1024}){
        int bl=(n+th-1)/th;cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);
        chain<<<bl,th>>>(x,n,iters);cudaDeviceSynchronize();
        cudaEventRecord(a);chain<<<bl,th>>>(x,n,iters);cudaEventRecord(b);cudaEventSynchronize(b);
        float ms;cudaEventElapsedTime(&ms,a,b);
        int blocksPerSm=0;cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSm,chain,th,0);
        std::cout<<"threads="<<th<<" active_blocks_per_sm="<<blocksPerSm<<" ms="<<ms<<"\n";
        cudaEventDestroy(a);cudaEventDestroy(b);
    }
    cudaFree(x);
}
