#include <cuda_runtime.h>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

__global__ void stride_read(const float* x,float* y,size_t n,int stride){
    size_t i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n){
        size_t j=(i*(size_t)stride)%n;
        y[i]=x[j]*1.0001f;
    }
}
int main(int argc,char** argv){
    size_t n=argc>1?std::strtoull(argv[1],nullptr,10):(1u<<24);
    std::vector<float> h(n,1.0f); float *dx,*dy;
    cudaMalloc(&dx,n*sizeof(float)); cudaMalloc(&dy,n*sizeof(float));
    cudaMemcpy(dx,h.data(),n*sizeof(float),cudaMemcpyHostToDevice);
    for(int stride: {1,2,4,8,16,32}){
        for(int w=0;w<3;++w) stride_read<<<(n+255)/256,256>>>(dx,dy,n,stride);
        cudaDeviceSynchronize();
        cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
        cudaEventRecord(a);
        stride_read<<<(n+255)/256,256>>>(dx,dy,n,stride);
        cudaEventRecord(b); cudaEventSynchronize(b);
        float ms=0; cudaEventElapsedTime(&ms,a,b);
        std::cout<<"stride="<<stride<<" ms="<<ms<<" effective_read_GBps="<<(n*sizeof(float)/1e6/ms)<<"\n";
        cudaEventDestroy(a); cudaEventDestroy(b);
    }
    cudaFree(dx); cudaFree(dy);
}
