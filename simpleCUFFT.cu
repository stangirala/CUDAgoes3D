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

void printData(Complex *a, int size, char *msg) {

  if (msg == "") printf("\n");
  else printf("%s\n", msg);

  for (int i = 0; i < size; i++)
    printf("%f %f\n", a[i].x, a[i].y);
}

void normData(Complex *a, int size, float norm) {

  for (int i = 0; i < size; i++) {
    a[i].x /= norm;
    a[i].y /= norm;
  }
}

// flag = 1 for real signals.
void randomFill(Complex *h_signal, int size, int flag) {

  // Real signal.
  if (flag == 1) {
    for (int i = 0; i < size; i++) {
      h_signal[i].x = rand() / (float) RAND_MAX;
      h_signal[i].y = 0;
    }
  }
}

// FFT a signal that's on the _DEVICE_.
void signalFFT(Complex *d_signal, int signal_size) {

  // Handle type used to store and execute CUFFT plans.
  // Essentially allocates the resouecwes and sort of interns
  // them.

  cufftHandle plan;
  cufftPlan1d(&plan, signal_size, CUFFT_C2C, 1);
  printf("Plan Created\n");

  // Execute the plan.
  cufftExecC2C(plan, (cufftComplex *) d_signal, (cufftComplex *) d_signal, CUFFT_FORWARD);
}

void signalIFFT(Complex *d_signal, int signal_size) {

  // Reverse of the signalFFT(.) function.

  cufftHandle plan;
  cufftPlan1d(&plan, signal_size, CUFFT_C2C, 1);
  printf("Plan Created\n");

  cufftExecC2C(plan, (cufftComplex *) d_signal, (cufftComplex *) d_signal, CUFFT_INVERSE);
}

int main()
{

  Complex *h_signal, *d_signal1, *d_signal2;

  int alloc_size = SIGNAL_SIZE;

  h_signal = (Complex *) malloc(sizeof(Complex) * alloc_size);

  cudaMalloc(&d_signal1, sizeof(Complex) * alloc_size);
  cudaMalloc(&d_signal2, sizeof(Complex) * alloc_size);

  // Add random data to signal.
  randomFill(h_signal, alloc_size, 1);
  printData(h_signal, alloc_size, "H1");
  cudaMemcpy(d_signal1, h_signal, sizeof(Complex) * alloc_size, cudaMemcpyHostToDevice);

  randomFill(h_signal, alloc_size, 1);
  printData(h_signal, alloc_size, "H2");
  cudaMemcpy(d_signal2, h_signal, sizeof(Complex) * alloc_size, cudaMemcpyHostToDevice);

  printf("MemCpy Fwd\n");

  signalFFT(d_signal1, alloc_size);
  signalIFFT(d_signal1, alloc_size);
  signalFFT(d_signal2, alloc_size);
  signalIFFT(d_signal2, alloc_size);

  printf("FFT Executed\n");

  cudaMemcpy(h_signal, d_signal2, sizeof(Complex) * alloc_size, cudaMemcpyDeviceToHost);

  printf("MemCpy Bakwd\n");

  normData(h_signal, alloc_size, 10);

  printData(h_signal, alloc_size, "D2 after FFT.");

  return 0;
}
