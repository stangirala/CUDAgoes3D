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

  /*h_signal[0].x = 0.0; h_signal[0].y = 0.0;
  h_signal[1].x = 0.0; h_signal[1].y = 0.0;
  h_signal[2].x = 0.0; h_signal[2].y = 0.0;
  h_signal[3].x = 0.0; h_signal[3].y = 0.0;
  h_signal[4].x = 1.0; h_signal[4].y = 0.0;
  h_signal[5].x = 2.0; h_signal[5].y = 0.0;
  h_signal[6].x = 3.0; h_signal[6].y = 0.0;
  h_signal[7].x = 4.0; h_signal[7].y = 0.0;
  h_signal[8].x = 5.0; h_signal[8].y = 0.0;
  h_signal[9].x = 4.0; h_signal[9].y = 0.0;
  h_signal[10].x = 3.0; h_signal[10].y = 0.0;
  h_signal[11].x = 2.0; h_signal[11].y = 0.0;
  h_signal[12].x = 1.0; h_signal[12].y = 0.0;
  h_signal[13].x = 0.0; h_signal[13].y = 0.0;
  h_signal[14].x = 0.0; h_signal[14].y = 0.0;
  h_signal[15].x = 0.0; h_signal[15].y = 0.0;*/

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
  cufftPlan1d(&plan, signal_size, CUFFT_C2C, 1);

  // Execute the plan.
  cufftExecC2C(plan, (cufftComplex *) d_signal, (cufftComplex *) d_signal, CUFFT_FORWARD);
}

void
signalIFFT(Complex *d_signal, int signal_size) {

  // Reverse of the signalFFT(.) function.

  cufftHandle plan;
  cufftPlan1d(&plan, signal_size, CUFFT_C2C, 10);
  //cufftPlan2d(&plan, signal_size, signal_size, CUFFT_R2C);

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


  /*int numThreads, threadID;
  numThreads = blockDim.x * gridDim.x;
  threadID = blockIdx.x * blockDim.x + threadIdx.x;

  for (int i = threadID; i < size1; i += numThreads) {
    signal1[i].x = signal1[i].x * signal2[i].x;
    signal1[i].y = signal1[i].y * signal2[i].y;
  }*/

}

void
cudaConvolution(Complex *d_signal1, int size1, Complex *d_signal2,
                int size2, dim3 blockSize, dim3 gridSize) {

  Complex *h_signal;
  int alloc_size = size1;

  h_signal = (Complex *) malloc(sizeof(Complex) * alloc_size);

  signalFFT(d_signal1, size1);
  cudaMemcpy(h_signal, d_signal1, sizeof(Complex) * alloc_size, cudaMemcpyHostToDevice);
  printData(h_signal, alloc_size, "1FFT");

  /*signalFFT(d_signal2, size2);
  cudaMemcpy(h_signal, d_signal2, sizeof(Complex) * alloc_size, cudaMemcpyHostToDevice);
  printData(h_signal, alloc_size, "2FFT");

  pwProd<<<gridSize, blockSize>>>(d_signal1, size1, d_signal2, size2);
  cudaMemcpy(h_signal, d_signal1, sizeof(Complex) * alloc_size, cudaMemcpyHostToDevice);
  printData(h_signal, alloc_size, "PwPrd");

  signalIFFT(d_signal1, size1);
  cudaMemcpy(h_signal, d_signal1, sizeof(Complex) * alloc_size, cudaMemcpyHostToDevice);
  printData(h_signal, alloc_size, "IFFT");*/

}


// Swap stuff to get the DIC FFT order correct.
/*__global__ void
swap(Complex *d_signal1, int size1) {
}*/

// factor represents how the DIC algorithm is applied on the convolution.
// That is, a factor of 16 implies a 16 way split on the signal and a 16 way
// call on the convolution.
// Assuming both signals are of the same size.
void
cudaConvolutionDIC(Complex *d_signal1, int size1, Complex *d_signal2, int size2, dim3 blockSize, dim3 gridSize, int load) {

  // TODO Padding!

  int i;

  if (load >= size1) {
    cudaConvolution(d_signal1, load, d_signal2, load, blockSize, gridSize);
  }
  else {
    // Wrong
    blockSize.x = load;
    for (i = 0; i < size1 / 2; i += 2)
      cudaConvolution((d_signal1 + i * load), load, (d_signal2 + i * load), load, blockSize, gridSize);
  }
}

int main()
{

  // #define alloc_size 10000

  Complex *h_signal, *d_signal1, *d_signal2;

  int alloc_size;

  alloc_size = 16;

  // Kernel Block and Grid Size.
  const dim3 blockSize(16, 16, 1);
  const dim3 gridSize(alloc_size / 16 + 1, alloc_size / 16 + 1, 1);

  h_signal = (Complex *) malloc(sizeof(Complex) * alloc_size);

  cudaMalloc(&d_signal1, sizeof(Complex) * alloc_size);
  cudaMalloc(&d_signal2, sizeof(Complex) * alloc_size);

  // Add random data to signal.
  randomFill(h_signal, alloc_size, REAL);

  printData(h_signal, alloc_size, "Random H1");
  cudaMemcpy(d_signal1, h_signal, sizeof(Complex) * alloc_size, cudaMemcpyHostToDevice);

  //randomFill(h_signal, alloc_size, REAL);
  /*printData(h_signal, alloc_size, "Random H2");
  cudaMemcpy(d_signal2, h_signal, sizeof(Complex) * alloc_size, cudaMemcpyHostToDevice);*/

  cudaConvolution(d_signal1, alloc_size, d_signal2, alloc_size, blockSize, gridSize);
  //cudaConvolutionDIC(d_signal1, alloc_size, d_signal2, alloc_size, blockSize, gridSize, 2);

  cudaDeviceSynchronize();

  /*cudaMemcpy(h_signal, d_signal1, sizeof(Complex) * alloc_size, cudaMemcpyDeviceToHost);
  //normData(h_signal, alloc_size, 10);
  printData(h_signal, alloc_size, "D1 * D2");*/

  return 0;
}
