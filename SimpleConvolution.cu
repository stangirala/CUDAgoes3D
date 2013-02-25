/*
* Simpleconvolution.cu
*
* procedure CUDA CONVOLUTION(signal, kernel, K, L, M, norm)
* cuMemcpy(gpu s, signal, HostToDevice)
* cuMemcpy(gpu k, kernel, HostToDevice)
* gpu s ← cuFFT(gpu s)
* gpu k ← cuFFT(gpu k)
* gpu s ← pwProd(gpu s, gpu k, K, L, M, norm)
* gpu s ← cuIFFT(gpu s)
* cuMemcpy(signal, gpu s, DeviceToHost)
* end procedure
*
*/


#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda_runtime.h>
#include <cufft.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#define SIGNAL_SIZE  10

typedef float2 Complex;

void printData(Complex *a, int size)
{
  for (int i = 0; i < size; i++)
    printf("%f %f\n", a[i].x, a[i].y);
}

int main()
{

  Complex *h_signal, *d_signal;

  int alloc_size = SIGNAL_SIZE;

  h_signal = (Complex *) malloc(alloc_size);

  cudaMalloc((void **)&d_signal, alloc_size);

  for (int i = 0; i < SIGNAL_SIZE; i++) {
    h_signal[i].x = rand() / (float) RAND_MAX;
    h_signal[i].y = rand() / (float) RAND_MAX;
  }

  printData(h_signal, 2);

  cudaMemcpy(d_signal, h_signal, alloc_size, cudaMemcpyHostToDevice);

  cufftHandle plan;
  cufftPlan1d(&plan, alloc_size, CUFFT_C2C, 1);

  cufftExecC2C(plan, (cufftComplex *) d_signal, (cufftComplex *) d_signal, CUFFT_FORWARD);

  cudaMemcpy(h_signal, d_signal, alloc_size, cudaMemcpyDeviceToHost);

  printData(h_signal, 2);

  return 0;
}
