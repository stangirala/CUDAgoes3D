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
//#include <helper_functions.h>
//#include <helper_cuda.h>

#define SIGNAL_SIZE  10

typedef float2 Complex;

void printData(Complex *a, int size)
{
  for (int i = 0; i < size; i++)
    printf("%f %f\n", a[i].x, a[i].y);
}

void normData(Complex *a, int size, float norm) {

  for (int i = 0; i < size; i++) {
    a[i].x /= norm;
    a[i].y /= norm;
  }
}

int main()
{

  Complex *h_signal, *d_signal;

  int alloc_size = SIGNAL_SIZE;

  h_signal = (Complex *) malloc(sizeof(Complex) * alloc_size);

  //checkCudaErrors(cudaMalloc(&d_signal, sizeof(Complex) * alloc_size));
  cudaMalloc(&d_signal, sizeof(Complex) * alloc_size);

  for (int i = 0; i < SIGNAL_SIZE; i++) {
    h_signal[i].x = rand() / (float) RAND_MAX;
    //h_signal[i].y = rand() / (float) RAND_MAX;
    h_signal[i].y = 0;
  }

  printData(h_signal, alloc_size);

  //checkCudaErrors(cudaMemcpy(d_signal, h_signal, sizeof(Complex) * alloc_size, cudaMemcpyHostToDevice));
  cudaMemcpy(d_signal, h_signal, sizeof(Complex) * alloc_size, cudaMemcpyHostToDevice);

  printf("MemCpy Fwd\n");

  // Handle type used to store and execute CUFFT plans.
  cufftHandle plan;
  //checkCudaErrors(cufftPlan1d(&plan, alloc_size, CUFFT_C2C, 1));
  cufftPlan1d(&plan, alloc_size, CUFFT_C2C, 1);

  //getLastCudaError("Kernel execution failed [ ComplexPointwiseMulAndScale ]");

  printf("Plan Created\n");

  //checkCudaErrors(cufftExecC2C(plan, (cufftComplex *) d_signal, (cufftComplex *) d_signal, CUFFT_FORWARD));
  cufftExecC2C(plan, (cufftComplex *) d_signal, (cufftComplex *) d_signal, CUFFT_FORWARD);

  //checkCudaErrors(cufftExecC2C(plan, (cufftComplex *) d_signal, (cufftComplex *) d_signal, CUFFT_INVERSE));
  cufftExecC2C(plan, (cufftComplex *) d_signal, (cufftComplex *) d_signal, CUFFT_INVERSE);

  printf("Executed\n");

  //checkCudaErrors(cudaMemcpy(h_signal, d_signal, sizeof(Complex) * alloc_size, cudaMemcpyDeviceToHost));
  cudaMemcpy(h_signal, d_signal, sizeof(Complex) * alloc_size, cudaMemcpyDeviceToHost);

  printf("MemCpy Bakwd\n");

  normData(h_signal, alloc_size, 10);

  printData(h_signal, alloc_size);

  return 0;
}
