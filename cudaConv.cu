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

#define SIGNAL_SIZE 10000 
#define FACTOR 4

typedef float2 Complex;

void
printData(Complex *a, int size, char *msg) {

  if (msg == "") printf("\n");
  else printf("%s\n", msg);

  for (int i = 0; i < size; i++)
    printf("%f %f\n", a[i].x, a[i].y);
}

void
normData(Complex *a, int size, float norm) {

  for (int i = 0; i < size; i++) {
    a[i].x /= norm;
    a[i].y /= norm;
  }
}

// flag = 1 for real signals.
void
randomFill(Complex *h_signal, int size, int flag) {

  // Real signal.
  if (flag == 1) {
    for (int i = 0; i < size; i++) {
      h_signal[i].x = rand() / (float) RAND_MAX;
      h_signal[i].y = 0;
    }
  }
}

// FFT a signal that's on the _DEVICE_.
void
signalFFT(Complex *d_signal, int signal_size) {

  // Handle type used to store and execute CUFFT plans.
  // Essentially allocates the resouecwes and sort of interns
  // them.

  cufftHandle plan;
  cufftPlan1d(&plan, signal_size, CUFFT_C2C, 1);

  // Execute the plan.
  cufftExecC2C(plan, (cufftComplex *) d_signal, (cufftComplex *) d_signal, CUFFT_FORWARD);
}

void
signalIFFT(Complex *d_signal, int signal_size) {

  // Reverse of the signalFFT(.) function.

  cufftHandle plan;
  cufftPlan1d(&plan, signal_size, CUFFT_C2C, 1);

  cufftExecC2C(plan, (cufftComplex *) d_signal, (cufftComplex *) d_signal, CUFFT_INVERSE);
}

// Pointwise Multiplication Kernel.
__global__ void
pwProd(Complex *signal1, int size1, Complex *signal2, int size2) {

  int threadsPerBlock, blockId, globalIdx;

  threadsPerBlock = blockDim.x * blockDim.y;
  blockId = blockIdx.x + (blockIdx.y * gridDim.x);
  globalIdx = (blockId * threadsPerBlock) + threadIdx.x + (threadIdx.y * blockDim.x);

  signal1[globalIdx].x = signal1[globalIdx].x * signal2[globalIdx].x;
  signal1[globalIdx].y = signal1[globalIdx].y * signal2[globalIdx].y;
}

void
cudaConvolution(Complex *d_signal1, int size1, Complex *d_signal2,
                int size2, const dim3 blockSize, const dim3 gridSize) {

  signalFFT(d_signal1, size1);

  signalFFT(d_signal2, size2);

  pwProd<<<blockSize, gridSize>>>(d_signal1, size1, d_signal2, size2);

  signalIFFT(d_signal1, size1);
}

// factor represents how the DIC algorithm is applied on the convolution.
// That is, a factor of 16 implies a 16 way split on the signal and a 16 way
// call on the convolution.
void
cudaConvolutionDIC(Complex *d_signal1, int size1, Complex *d_signal2,
                   int size2, const dim3 blockSize, const dim3 gridSize, int factor) {

  int load, i;

  // DIC? What DIC?
  if (factor >= size1) {
    cudaConvolution(d_signal1, size1, d_signal2, size2, blockSize, gridSize);
  }
  else {
    load = size1 / factor;
    for (i = 0; i < load; i++)
      cudaConvolution((d_signal1 + i * load), load, (d_signal2 + i * load), load, blockSize, gridSize);
  }
}

int main()
{

  Complex *h_signal, *d_signal1, *d_signal2;

  int alloc_size, numRows, numCols;

  alloc_size = SIGNAL_SIZE;

  // This random case! Image paremeters.
  numRows = alloc_size;
  numCols = alloc_size;

  // Kernel Block and Grid Size.
  const dim3 blockSize(alloc_size, alloc_size, 1);
  const dim3 gridSize( numRows / alloc_size + 1, numCols / alloc_size + 1, 1);
  //const dim3 blockSize(4, 4, 1);
  //const dim3 gridSize( numRows / 4 + 1, numCols / 4 + 1, 1);

  h_signal = (Complex *) malloc(sizeof(Complex) * alloc_size);

  cudaMalloc(&d_signal1, sizeof(Complex) * alloc_size);
  cudaMalloc(&d_signal2, sizeof(Complex) * alloc_size);

  // Add random data to signal.
  randomFill(h_signal, alloc_size, 1);
  printData(h_signal, alloc_size, "Random H1");
  cudaMemcpy(d_signal1, h_signal, sizeof(Complex) * alloc_size, cudaMemcpyHostToDevice);

  randomFill(h_signal, alloc_size, 1);
  printData(h_signal, alloc_size, "Random H2");
  cudaMemcpy(d_signal2, h_signal, sizeof(Complex) * alloc_size, cudaMemcpyHostToDevice);

  cudaMemcpy(h_signal, d_signal1, sizeof(Complex) * alloc_size, cudaMemcpyDeviceToHost);
  //printData(h_signal, alloc_size, "H1 FFT");

  cudaMemcpy(h_signal, d_signal2, sizeof(Complex) * alloc_size, cudaMemcpyDeviceToHost);
  //printData(h_signal, alloc_size, "H2 FFT");

  //cudaConvolution(d_signal1, alloc_size, d_signal2, alloc_size, blockSize, gridSize);
  //cudaConvolution(d_signal1, 4, d_signal2, 4, blockSize, gridSize);
  cudaConvolutionDIC(d_signal1, alloc_size, d_signal2, alloc_size, blockSize, gridSize, alloc_size);

  cudaMemcpy(h_signal, d_signal1, sizeof(Complex) * alloc_size, cudaMemcpyDeviceToHost);
  normData(h_signal, alloc_size, 10);
  printData(h_signal, alloc_size, "D1 * D2");

  return 0;
}
