#include <cudaConv.h>

void Odeenfft() {

  Complex *h1, *h2, *d1, *d2;

  int s1, s2, newsize;


  s1 = 16; s2 = 16;

  newsize = allocateAndPad1D(&h1, s1, &h2, s2);

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

  cudaMemcpy(h1, d1, sizeof(Complex) * newsize, cudaMemcpyDeviceToHost);

  printDeviceData(d1, newsize, "D1");

  normData(h1, newsize, newsize);

  printHostData(h1, newsize, "1D Convolution");

  free(h1); free(h2);
  cudaFree(d1); cudaFree(d2);

}

void Trittfft() {

  Complex *h1, *d1, *h2, *d2;

  int s1, s2, newsize;


  s1 = 16;
  s2 = 16;

  newsize = allocateAndPad3D(&h1, s1, &h2, s2);

  randomFill(h1, s1 + s1 + s1, REAL);
  printHostData(h1, s1 + s1 + s1, "h1");
  cudaMalloc(&d1, sizeof(Complex) * newsize * newsize * newsize);
  cudaMemcpy(d1, h1, sizeof(Complex) * newsize * newsize * newsize, cudaMemcpyHostToDevice);

  randomFill(h2, s2 + s2 + s2, REAL);
  printHostData(h2, s2 + s2 + s2, "h2");
  cudaMalloc(&d2, sizeof(Complex) * newsize * newsize * newsize);
  cudaMemcpy(d2, h2, sizeof(Complex) * newsize * newsize * newsize, cudaMemcpyHostToDevice);

  // Kernel Block and Grid Size.
  const dim3 blockSize(16, 16, 1);
  const dim3 gridSize((newsize) / 16 + 1, (newsize) / 16 + 1, 1);

  cudaConvolution3D(d1, newsize, d2, newsize, blockSize, gridSize);

  //normData(h1, (newsize + newsize + newsize), newsize * newsize * newsize);

  printHostData(h1, s1 + s1 + s1, "3D Convolution");
  printf ("\n");

  free(h1); free(h2);
  cudaFree(d1); cudaFree(d2);

}

int main()
{

  if (checkCUDA()) {
    printf ("CUDA FAIL\n");
    exit(1);
  }

  //cudaConvInit();

  Odeenfft();

  cudaDeviceReset();

  //Trittfft();

  return 0;
}
