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


  dim = 1;

  for (i = 0; i < dim; i++)  {

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

  return 0;
}
