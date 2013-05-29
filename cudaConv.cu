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
#include <cuda.h>

typedef enum signaltype {REAL, COMPLEX} signal;

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
  if (flag == REAL) {
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
  if (cufftPlan1d(&plan, signal_size, CUFFT_C2C, 1) != CUFFT_SUCCESS) {
    printf("Failed to plan FFT\n");
    exit(0);
  }

  // Execute the plan.
  if (cufftExecC2C(plan, (cufftComplex *) d_signal, (cufftComplex *) d_signal, CUFFT_FORWARD) != CUFFT_SUCCESS) {
    printf ("Failed Executing FFT\n");
    exit(0);
  }

}


// Reverse of the signalFFT(.) function.
void
signalIFFT(Complex *d_signal, int signal_size) {

  cufftHandle plan;
  if (cufftPlan1d(&plan, signal_size, CUFFT_C2C, 1) != CUFFT_SUCCESS) {
    printf("Failed to plan IFFT\n");
    exit(0);
  }

  if (cufftExecC2C(plan, (cufftComplex *) d_signal, (cufftComplex *) d_signal, CUFFT_INVERSE) != CUFFT_SUCCESS) {
    printf ("Failed Executing FFT\n");
    exit(0);
  }
}


// Pointwise Multiplication Kernel.
__global__ void
pwProd(Complex *signal1, int size1, Complex *signal2, int size2) {

  int threadsPerBlock, blockId, globalIdx;

  threadsPerBlock = blockDim.x * blockDim.y;
  blockId = blockIdx.x + (blockIdx.y * gridDim.x);
  globalIdx = (blockId * threadsPerBlock) + threadIdx.x + (threadIdx.y * blockDim.x);

  if (globalIdx < size1) {
      signal1[globalIdx].x = signal1[globalIdx].x * signal2[globalIdx].x - signal1[globalIdx].y * signal2[globalIdx].y;
      signal1[globalIdx].y = signal1[globalIdx].x * signal2[globalIdx].y + signal1[globalIdx].y * signal2[globalIdx].x;
    }

}

void
cudaConvolution(Complex *d_signal1, int size1, Complex *d_signal2,
                int size2, dim3 blockSize, dim3 gridSize) {

  signalFFT(d_signal1, size1);
  signalFFT(d_signal2, size2);

  pwProd<<<gridSize, blockSize>>>(d_signal1, size1, d_signal2, size2);

  signalIFFT(d_signal1, size1);

}


void
cudaConvolutionDIC(Complex *h1, int size1, Complex *h2, int size2, dim3 blockSize, dim3 gridSize) {

  // TODO Padding!

  int i, alloc_size = 16;

  Complex *d_signal1, *d_signal2, temp;

  cudaMalloc(&d_signal1, sizeof(Complex) * alloc_size);
  cudaMalloc(&d_signal2, sizeof(Complex) * alloc_size);

  for (i = 0; i < size1/2; i++) {
    h1[i].x = (h1[i + size2 / 2].x + h1[i].x);
    h1[i].y = (h1[i + size2 / 2].y + h1[i].y);
  }
  for (i = size1/2; i < size1; i++) {
    h1[i].x = (h1[i].x - h1[i - size2 / 2].x) * exp(-2 * 3.14159 * (size1/2 - i)/size1);
    h1[i].y = (h1[i].y - h1[i - size2 / 2].y) * exp(-2 * 3.14159 * (size1/2 - i)/size1);
  }

  for (i = 0; i < size2/2; i++) {
    h2[i].x = (h2[i + size2 / 2].x + h2[i].x);
    h2[i].y = (h2[i + size2 / 2].y + h2[i].y);
  }
  for (i = size2/2; i < size1/2; i++) {
    h2[i].x = (h2[i].x - h2[i - size2 / 2].x) * exp(-2 * 3.14159 * (size1/2 - i)/size2);
    h2[i].y = (h2[i].y - h2[i - size2 / 2].y) * exp(-2 * 3.14159 * (size1/2 - i)/size2);
  }

  cudaMemcpy(d_signal1, h1, sizeof(Complex) * size1, cudaMemcpyHostToDevice);
  cudaMemcpy(d_signal2, h2, sizeof(Complex) * size2, cudaMemcpyHostToDevice);

  cudaConvolution(d_signal1, size1/2, d_signal2, size2/2, blockSize, gridSize);
  cudaConvolution(d_signal1 + size1/2, size1/2, d_signal2 + size2/2, size2/2, blockSize, gridSize);

  cudaMemcpy(h1, d_signal1, sizeof(Complex) * size1, cudaMemcpyDeviceToHost);

  for (i = 1; i < size1 / 2 - 1; i += 2) {

    temp = h1[i];
    h1[i] = h1[i + size1 / 2 - 1];
    h1[i + size1 / 2] = temp;
  }

}

void allocateAndPad(Complex **a, int size) {

  int oldsize = size, i;

  while (!((size != 0) && !(size & (size - 1)))) {
    size++;
  }

  *a = (Complex *) malloc(sizeof(Complex) * size);
  for (i = oldsize; i < size; i++) {
    (*a)[i].x = 0;
    (*a)[i].y = 0;
  }
}

int main()
{

  Complex *h_signal, *d_signal1, *d_signal2, *h1, *h2;

  int alloc_size, i, type, dim;

  alloc_size = 16;

  int deviceCount;
  cudaError_t e = cudaGetDeviceCount(&deviceCount);
  if (e != cudaSuccess) {
    return -1;
  }

  // Kernel Block and Grid Size.
  const dim3 blockSize(16, 16, 1);
  const dim3 gridSize(alloc_size / 16 + 1, alloc_size / 16 + 1, 1);

  h1 = (Complex *) malloc(sizeof(Complex) * alloc_size);
  h2 = (Complex *) malloc(sizeof(Complex) * alloc_size);


  type = 0;
  dim = 1;

  for (i = 0; i < dim; i++)  {

    if (type == 1) {

      allocateAndPad(&h_signal, 16);

      randomFill(h_signal, 16, REAL);

      printData(h_signal, alloc_size, "H Signal 1");
      printData(h_signal, alloc_size, "H Signal 2");

      cudaMalloc(&d_signal1, sizeof(Complex) * alloc_size);
      cudaMalloc(&d_signal2, sizeof(Complex) * alloc_size);
      cudaMemcpy(d_signal1, h_signal, sizeof(Complex) * alloc_size, cudaMemcpyHostToDevice);
      cudaMemcpy(d_signal2, h_signal, sizeof(Complex) * alloc_size, cudaMemcpyHostToDevice);

      cudaConvolution(d_signal1, alloc_size, d_signal2, alloc_size, blockSize, gridSize);

      cudaDeviceSynchronize();

      cudaMemcpy(h_signal, d_signal1, sizeof(Complex) * alloc_size, cudaMemcpyDeviceToHost);

      normData(h_signal, alloc_size, alloc_size);

      printData(h_signal, alloc_size, "Conv");
    }
    else {

      randomFill(h1, alloc_size, REAL);
      randomFill(h2, alloc_size, REAL);

      printData(h1, alloc_size, "H1");
      printData(h2, alloc_size, "H2");

      cudaConvolutionDIC(h1, alloc_size, h2, alloc_size, blockSize, gridSize);
      cudaDeviceSynchronize();
      normData(h1, alloc_size, alloc_size);
      printData(h1, alloc_size, "Conv");
    }

  }

  return 0;
}
