#include <cudaConv.h>

int main()
{

  Complex *h1, *h2, *d1, *d2;

  int s1, s2, newsize;


  if (checkCUDA()) {
    printf ("CUDA FAIL\n");
    exit(1);
  }

  cudaConvInit();

  s1 = 16 + 16 + 16;
  s2 = 16 + 16 + 16;
  //s1 = 16; s2 = 16;

  newsize = allocateAndPad(&h1, s1, &h2, s2);

  randomFill(h1, s1, REAL);
  randomFill(h2, s2, REAL);
  printHostData(h1, newsize, "h1");
  printHostData(h2, newsize, "h2");

  // Kernel Block and Grid Size.
  const dim3 blockSize(16, 16, 1);
  const dim3 gridSize(newsize / 16 + 1, newsize / 16 + 1, 1);

  cudaMalloc(&d1, sizeof(Complex) * newsize);
  cudaMalloc(&d2, sizeof(Complex) * newsize);

  cudaMemcpy(d1, h1, sizeof(Complex) * newsize, cudaMemcpyHostToDevice);
  cudaMemcpy(d2, h2, sizeof(Complex) * newsize, cudaMemcpyHostToDevice);

  cudaConvolution1D(d1, newsize, d2, newsize, blockSize, gridSize);

  //signalFFT3D(d1, 16, 16, 16);

  cudaMemcpy(h1, d1, sizeof(Complex) * newsize, cudaMemcpyDeviceToHost);

  normData(h1, newsize, newsize);

  printHostData(h1, newsize, "h1 3D FFT");

  free(h1); free(h2);
  cudaFree(d1); cudaFree(d2);

  cudaDeviceReset();

  return 0;
}
