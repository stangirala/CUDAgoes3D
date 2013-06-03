#include <cudaConv.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda_runtime.h>
#include <cufft.h>
#include <cuda.h>


int checkCUDA() {

  if ((system("nvidia-settings -q gpus >/dev/null")) == 0) {

    int deviceCount;
    cudaError_t e = cudaGetDeviceCount(&deviceCount);
    if (e != cudaSuccess) {
      return 1;
    }
  }
  else
    return 1;

  return 0;
}


void
printHostData(Complex *a, int size, char *msg) {

  if (msg == "") printf("\n");
  else printf("%s\n", msg);

  for (int i = 0; i < size; i++)
    printf("%f %f\n", a[i].x, a[i].y);
}


void
printDeviceData(Complex *a, int size, char *msg) {

  Complex *h;

  h = (Complex *) malloc(sizeof(Complex) * size);

  if (msg == "") printf("\n");
  else printf("%s\n", msg);

  cudaMemcpy(h, a, sizeof(Complex) * size, cudaMemcpyDeviceToHost);

  for (int i = 0; i < size; i++)
    printf("%f %f\n", h[i].x, h[i].y);
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

void
zeroFill(Complex *h_signal, int size) {

  for (int i = 0; i < size; i++) {
    h_signal[i].x = 0;
    h_signal[i].y = 0;
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

      signal1[globalIdx].x = (signal1[globalIdx].x * signal2[globalIdx].x) - (signal1[globalIdx].y * signal2[globalIdx].y);
      signal1[globalIdx].y = (signal1[globalIdx].x * signal2[globalIdx].y) + (signal1[globalIdx].y * signal2[globalIdx].x);
    }


  /*const int numThreads = blockDim.x * gridDim.x;
  const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = threadID; i < size1; i += numThreads)
  {
    signal1[i].x = (signal1[i].x * signal2[i].x - signal1[i].y * signal2[i].y);
    signal1[i].y = (signal1[i].x * signal2[i].y + signal1[i].y * signal2[i].x);
  }*/

}

void
cudaConvolution(Complex *d_signal1, int size1, Complex *d_signal2,
                int size2, dim3 blockSize, dim3 gridSize) {

  //Complex *h1;
  //h1 = (Complex *)malloc(sizeof(Complex) * size1);

  signalFFT(d_signal1, size1);
  signalFFT(d_signal2, size2);

  printDeviceData(d_signal1, size1, "H1 FFT");
  printDeviceData(d_signal2, size2, "H2 FFT");
  pwProd<<<gridSize, blockSize>>>(d_signal1, size1, d_signal2, size2);
  printDeviceData(d_signal1, size1, "PwProd");


/*h1[0].x = 64.7652;      h1[0].y = 0;
h1[1].x = -20.0979;     h1[1].y = -1.7253;
h1[2].x = 1.7976;       h1[2].y = 0.8094;
h1[3].x = -5.1845;      h1[3].y = -2.5464;
h1[4].x = 0.1483;       h1[4].y = 0.1457;
h1[5].x = -3.4572;      h1[5].y = -2.5441;
h1[6].x = 0.8347;       h1[6].y = 0.3288;
h1[7].x = -0.4145;      h1[7].y = -0.1047;
h1[8].x = -1.2685;      h1[8].y = 1.1672;
h1[9].x = 1.6341;       h1[9].y = -2.7998;
h1[10].x = 0.5429;      h1[10].y = -0.0323;
h1[11].x = 0.2442;      h1[11].y = -2.5289;
h1[12].x = 0.6802;      h1[12].y = 0.1128;
h1[13].x = 0.9380;      h1[13].y = 1.6687;
h1[14].x = 1.3181;      h1[14].y = -1.7602;
h1[15].x = -1.4486;     h1[15].y = -1.0052;
h1[16].x = -0.2072;     h1[16].y = 0;
h1[17].x = -1.4486;     h1[17].y = 1.0052;
h1[18].x = 1.3181;      h1[18].y = 1.7602;
h1[19].x = 0.9380;      h1[19].y = -1.6687;
h1[20].x = 0.6802;      h1[20].y = -0.1128;
h1[21].x = 0.2442;      h1[21].y = 2.5289;
h1[22].x = 0.5429;      h1[22].y = 0.0323;
h1[23].x = 1.6341;      h1[23].y = 2.7998;
h1[24].x = -1.2685;     h1[24].y = -1.1672;
h1[25].x = -0.4145;     h1[25].y = 0.1047;
h1[26].x = 0.8347;      h1[26].y = -0.3288;
h1[27].x = -3.4572;     h1[27].y = 2.5441;
h1[28].x = 0.1483;      h1[28].y = -0.1457;
h1[29].x = -5.1845;     h1[29].y = 2.5464;
h1[30].x = 1.7976;      h1[30].y = -0.8094;
h1[31].x = -20.0979;    h1[31].y = 1.7253;*/


  //cudaMemcpy(d_signal1, h1, sizeof(Complex) * size1, cudaMemcpyHostToDevice);

  //signalIFFT(d_signal1, size1);

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
  *b = (Complex *) malloc(sizeof(Complex) * newsize);
  for (i = oldsize; i < newsize; i++) {
    (*b)[i].x = 0;
    (*b)[i].y = 0;
  }

  return newsize;
}

int main()
{

  Complex *h1, *h2, *d1, *d2;

  int s1, s2, newsize, i, dim;


  if (checkCUDA()) {
    printf ("CUDA FAIL\n");
    exit(0);
  }


  dim = 1;

  s1 = 16;
  s2 = 16;

  for (i = 0; i < dim; i++)  {

      newsize = allocateAndPad(&h1, s1, &h2, s2);

      randomFill(h1, s1, REAL);
      randomFill(h2, s2, REAL);

      // Kernel Block and Grid Size.
      const dim3 blockSize(16, 16, 1);
      const dim3 gridSize(newsize / 16 + 1, newsize / 16 + 1, 1);

      //printHostData(h1, newsize, "H Signal 1");
      //printHostData(h2, newsize, "H Signal 2");

      cudaMalloc(&d1, sizeof(Complex) * newsize);
      cudaMalloc(&d2, sizeof(Complex) * newsize);

      cudaMemcpy(d1, h1, sizeof(Complex) * newsize, cudaMemcpyHostToDevice);
      cudaMemcpy(d2, h2, sizeof(Complex) * newsize, cudaMemcpyHostToDevice);

      cudaConvolution(d1, newsize, d2, newsize, blockSize, gridSize);

      //cudaDeviceSynchronize();

      cudaMemcpy(h1, d1, sizeof(Complex) * newsize, cudaMemcpyDeviceToHost);

      normData(h1, newsize, newsize);

      //printHostData(h1, newsize, "Conv");

      free(h1); free(h2);
      cudaFree(d1); cudaFree(d2);

      cudaDeviceReset();
  }

  return 0;
}
