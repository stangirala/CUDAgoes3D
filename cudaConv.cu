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


int allocateAndPad(Complex **a, int s1, Complex **b, int s2) {

  int oldsize, newsize, i;

  newsize = s1 + s2 - 1;

  while (!((newsize != 0) && !(newsize & (newsize - 1)))) {
    newsize++;
  }

  oldsize = s1;
  *a = (Complex *) malloc(sizeof(Complex) * newsize);
  for (i = oldsize; i < newsize; i++) {
    (*a)[i].x = 0;
    (*a)[i].y = 0;
  }

  oldsize = s2;
  *b = (Complex *) malloc(sizeof(Complex) * s2);
  for (i = oldsize; i < newsize; i++) {
    (*b)[i].x = 0;
    (*b)[i].y = 0;
  }

  return newsize;
}

int main()
{

  Complex *h1, *h2, *d_signal1, *d_signal2;

  cudaError_t error;

  int s1, s2, newsize, i, dim;

  int deviceCount;
  cudaError_t e = cudaGetDeviceCount(&deviceCount);
  if (e != cudaSuccess) {
    return -1;
  }

  dim = 1;

  s1 = 16;
  s2 = 16;

  for (i = 0; i < dim; i++)  {

      newsize = allocateAndPad(&h1, s1, &h2, s2);
      randomFill(h1, s1, REAL);
      randomFill(h2, s2, REAL);

      //h1 = (Complex *) malloc(sizeof(Complex) * s1);
      //h2 = (Complex *) malloc(sizeof(Complex) * s2);
      //newsize = 16;
      //randomFill(h1, s1, REAL);
      //randomFill(h2, s2, REAL);

      // Kernel Block and Grid Size.
      const dim3 blockSize(16, 16, 1);
      const dim3 gridSize(newsize / 16 + 1, newsize / 16 + 1, 1);

      printData(h1, newsize, "H Signal 1");
      printData(h2, newsize, "H Signal 2");

      printf("Done printing\n");

      cudaMalloc(&d_signal1, sizeof(Complex) * newsize);
      printf ("Allocating d h1\n");
      if ((error = cudaGetLastError()) != cudaSuccess) {
        printf ("Cuda Error: %s", cudaGetErrorString(error));
      }
      cudaMalloc(&d_signal2, sizeof(Complex) * newsize);
      if ((error = cudaGetLastError()) != cudaSuccess) {
        printf ("Cuda Error: %s", cudaGetErrorString(error));
      }

      printf("Trying to copy to device\n");

      cudaMemcpy(d_signal1, h1, sizeof(Complex) * newsize, cudaMemcpyHostToDevice);
      cudaMemcpy(d_signal2, h2, sizeof(Complex) * newsize, cudaMemcpyHostToDevice);

      printf("Trying to convolve\n");

      cudaConvolution(d_signal1, newsize, d_signal2, newsize, blockSize, gridSize);

      cudaDeviceSynchronize();

      cudaMemcpy(h1, d_signal1, sizeof(Complex) * newsize, cudaMemcpyDeviceToHost);

      normData(h1, newsize, newsize);

      printData(h1, newsize, "FFT H1");
  }

  return 0;
}
