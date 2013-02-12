/*
* Simpleconvolution.cu
*
*   procedure CUDA CONVOLUTION(signal, kernel, K, L, M, norm)
*     cuMemcpy(gpu s, signal, HostToDevice)
*     cuMemcpy(gpu k, kernel, HostToDevice)
*     gpu s ← cuFFT(gpu s)
*     gpu k ← cuFFT(gpu k)
*     gpu s ← pwProd(gpu s, gpu k, K, L, M, norm)
*     gpu s ← cuIFFT(gpu s)
*     cuMemcpy(signal, gpu s, DeviceToHost)
*   end procedure
*
*/

#include <stdio.h>
#include <cuda.h>

#include <cufft.h>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>


#define BATCH 10

int main()
{

  int *k_h, *s_d, *k_d;

  cufftComplex *s_h;

  int size;

  cufftComplex *data;

  // xyz dimension size of the sum of signals.
  int K, L, M;

  printf("Enter K, L, M\n");
  scanf("%d %d %d", &K, &L, &M);

  size = (sizeof(int) * (K + L + M));

  // Allocate host memory.
  /*s_h = (int *) malloc(sizeof(int) * size);
  k_h = (int *) malloc(sizeof(int) * size);

  // Temp.
  int k, l, m;

  // Generate both signal and kernel for now.
  for (k = 0; k < K; k++)
    for (l = 0; l < L; l++)
      for (m = 0; m < M; m++) {
        // First fill x, then y and then z.
        s_h[k + l + m] = m;
        k_h[k + l + m] = m;
      }*/

  s_h = (cufftComplex *) malloc(sizeof(cufftComplex) * K * BATCH);

  // Allocate Device Memory.
  cudaMalloc((void **) &s_d, size);
  cudaMalloc((void **) &data, (sizeof(cufftComplex) * K * BATCH)); // ComplexData
  cudaMalloc((void **) &k_d, size);

  // Copy to device.
  cudaMemcpy(s_d, s_h, size, cudaMemcpyHostToDevice);
  cudaMemcpy(k_d, k_h, size, cudaMemcpyHostToDevice);

  cufftHandle plan = 1;

  // Do an Dimension first. DFT size of K.

  for (int i = 0; i < (sizeof(cufftComplex) * K * BATCH); i++) {
    s_h[i].x = i;
    s_h[i].y = i;
  }

  cudaMemcpy(data, s_h, (sizeof(cufftComplex) * K * BATCH), cudaMemcpyHostToDevice);

  cufftPlan1d(&plan, K, CUFFT_C2C, BATCH);

  cufftExecC2C(plan, data, data, CUFFT_FORWARD);

  cudaThreadSynchronize();

  cufftDestroy(plan);

  cudaMemcpy(s_h, data, (sizeof(cufftComplex) * K * BATCH), cudaMemcpyDeviceToHost);

  cudaFree(data);

  for (int i = 0; i < (sizeof(cufftComplex) * K * BATCH); i++)
    printf("%f %f \n", s_h[i].x, s_h[i].y);

  // Copy results, back, from the GPU to the CPU.
  cudaMemcpy(s_h, s_d, size, cudaMemcpyDeviceToHost);

  // Cleanup
  free(s_h);
  free(k_h);
  cudaFree(s_d);
  cudaFree(k_d);

}
