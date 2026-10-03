// Build: nvcc -O3 examples/shared_memory_reuse.cu -o shared_memory_reuse
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <vector>
__global__ void neighbor_global(const float*x,float*y,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n){float a=x[i];float b=i?x[i-1]:0;float c=i+1<n?x[i+1]:0;y[i]=a+b+c;}
}
__global__ void neighbor_shared(const float*x,float*y,int n){
    extern __shared__ float s[]; int t=threadIdx.x,i=blockIdx.x*blockDim.x+t;
    if(i<n)s[t+1]=x[i];
    if(t==0)s[0]=(i>0)?x[i-1]:0;
    if(t==blockDim.x-1)s[t+2]=(i+1<n)?x[i+1]:0;
    __syncthreads();
    if(i<n)y[i]=s[t]+s[t+1]+s[t+2];
}
int main(){
    int n=1<<20;std::vector<float>h(n,1),a(n),b(n);float *x,*y;
    cudaMalloc(&x,n*4);cudaMalloc(&y,n*4);cudaMemcpy(x,h.data(),n*4,cudaMemcpyHostToDevice);
    int th=256,bl=(n+th-1)/th;
    neighbor_global<<<bl,th>>>(x,y,n);cudaMemcpy(a.data(),y,n*4,cudaMemcpyDeviceToHost);
    neighbor_shared<<<bl,th,(th+2)*sizeof(float)>>>(x,y,n);cudaMemcpy(b.data(),y,n*4,cudaMemcpyDeviceToHost);
    for(int i=0;i<n;++i)if(std::fabs(a[i]-b[i])>1e-5){std::cerr<<"mismatch\n";return 2;}
    std::cout<<"PASS. Profile both kernels to compare global traffic and resource use.\n";cudaFree(x);cudaFree(y);
}
