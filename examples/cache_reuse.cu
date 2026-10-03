// Build: nvcc -O3 examples/cache_reuse.cu -o cache_reuse
#include <cuda_runtime.h>
#include <iostream>
__global__ void reuse_kernel(const float*x,float*y,int n,int repeats){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n){
        float v=x[i];
        float acc=0;
        for(int r=0;r<repeats;++r) acc += v * (1.0f + 1e-6f*r);
        y[i]=acc;
    }
}
int main(){
    int n=1<<24;float *x,*y;cudaMalloc(&x,(size_t)n*4);cudaMalloc(&y,(size_t)n*4);cudaMemset(x,1,(size_t)n*4);
    for(int repeats:{1,4,16,64}){
        cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);
        reuse_kernel<<<(n+255)/256,256>>>(x,y,n,repeats);cudaDeviceSynchronize();
        cudaEventRecord(a);reuse_kernel<<<(n+255)/256,256>>>(x,y,n,repeats);cudaEventRecord(b);cudaEventSynchronize(b);
        float ms;cudaEventElapsedTime(&ms,a,b);
        std::cout<<"repeats="<<repeats<<" ms="<<ms<<"\n";
        cudaEventDestroy(a);cudaEventDestroy(b);
    }
    std::cout<<"Use Nsight Compute to observe the transition from memory-dominated toward compute-dominated work.\n";
    cudaFree(x);cudaFree(y);
}
