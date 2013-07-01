#include <cudaConv.h>

void Odeenfft() {

  Complex *h1, *h2, *d1, *d2;

  int s1, s2, newsize;


  s1 = 16; s2 = 16;

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

  cudaMemcpy(h1, d1, sizeof(Complex) * newsize, cudaMemcpyDeviceToHost);

  normData(h1, newsize, newsize);

  printHostData(h1, newsize, "h1 3D FFT");

  free(h1); free(h2);
  cudaFree(d1); cudaFree(d2);

}

void trittfft() {

  Complex *h1, *d1, *th1, *th2, *th3;

  int s1;


  s1 = 16;

  th1 = (Complex *) malloc(sizeof(Complex) * s1);
  th2 = (Complex *) malloc(sizeof(Complex) * s1);
  th3 = (Complex *) malloc(sizeof(Complex) * s1);

  h1 = (Complex *) malloc(sizeof(Complex) * (3 * s1));

  randomFill(th1, s1, REAL);
  printHostData(th1, s1, "th1");

  randomFill(th2, s1, REAL);
  printHostData(th2, s1, "th2");

  randomFill(th3, s1, REAL);
  printHostData(th3, s1, "th3");

  memcpy(h1, th1, sizeof(Complex) * s1);
  memcpy((h1 + s1), th2, sizeof(Complex) * s1);
  memcpy((h1 + 2*s1), th3, sizeof(Complex) * s1);
  printHostData(h1, 3*s1, "h1");

  cudaMalloc(&d1, sizeof(Complex) * (3 * s1));

  cudaMemcpy(d1, th1, sizeof(Complex) * s1, cudaMemcpyHostToDevice);
  cudaMemcpy((d1 + s1), th2, sizeof(Complex) * s1, cudaMemcpyHostToDevice);
  cudaMemcpy((d1 + 2*s1), th3, sizeof(Complex) * s1, cudaMemcpyHostToDevice);


  // Kernel Block and Grid Size.
  const dim3 blockSize(16, 16, 1);
  const dim3 gridSize((3 * s1) / 16 + 1, (3 * s1) / 16 + 1, 1);


  signalFFT3D(d1, s1, s1, s1);

  cudaMemcpy(h1, d1, sizeof(Complex) * (3 * s1), cudaMemcpyDeviceToHost);

  printHostData(h1, 3 * s1, "h1 3D FFT");

  free(h1);
  cudaFree(d1);

}

int main()
{

  if (checkCUDA()) {
    printf ("CUDA FAIL\n");
    exit(1);
  }

  cudaConvInit();

  //Odeenfft();

  cudaDeviceReset();

  trittfft();

  return 0;
}
