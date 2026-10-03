// Build: nvcc -O3 examples/bank_conflicts.cu -o bank_conflicts
#include <cuda_runtime.h>
#include <iostream>
template<int PAD>
__global__ void transpose_tile(float*out,const float*in,int n){
    __shared__ float tile[32][32+PAD];
    int x=blockIdx.x*32+threadIdx.x,y=blockIdx.y*32+threadIdx.y;
    if(x<n&&y<n)tile[threadIdx.y][threadIdx.x]=in[y*n+x];
    __syncthreads();
    x=blockIdx.y*32+threadIdx.x;y=blockIdx.x*32+threadIdx.y;
    if(x<n&&y<n)out[y*n+x]=tile[threadIdx.x][threadIdx.y];
}
int main(){
    int n=2048;float *a,*b;cudaMalloc(&a,(size_t)n*n*4);cudaMalloc(&b,(size_t)n*n*4);
    dim3 th(32,8),bl((n+31)/32,(n+31)/32);
    transpose_tile<0><<<bl,th>>>(b,a,n);cudaDeviceSynchronize();
    transpose_tile<1><<<bl,th>>>(b,a,n);cudaDeviceSynchronize();
    std::cout<<"Run Nsight Compute on kernels with PAD=0 and PAD=1; compare shared-memory bank-conflict metrics.\n";
    cudaFree(a);cudaFree(b);
}
