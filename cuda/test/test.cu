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

  normData(h1, newsize, newsize);

  printHostData(h1, newsize, "h1 3D FFT");

  free(h1); free(h2);
  cudaFree(d1); cudaFree(d2);

}

void trittfft() {

  Complex *h1, *d1, *h2, *d2, temp;

  int s1, s2, newsize, i;


  s1 = 4;
  s2 = 4;

  newsize = allocateAndPad3D(&h1, s1, &h2, s2);
  printf ("newsize is %d\n", newsize);

  //randomFill(h1, newsize + newsize + newsize, REAL);
  randomFill(h1, s1 + s1 + s1, REAL);
  //randomFill(h2, newsize + newsize + newsize, REAL);
  randomFill(h2, s2 + s2 + s2, REAL);

  //printHostData(h1, newsize + newsize + newsize, "h1");
  printHostData(h1, s1 + s1 + s1, "h1");
  //printHostData(h2, newsize + newsize + newsize, "h2");
  printHostData(h2, s2 + s2 + s2, "h2");

  cudaMalloc(&d1, sizeof(Complex) * newsize * newsize * newsize);

  cudaMemcpy(d1, h1, sizeof(Complex) * newsize * newsize * newsize, cudaMemcpyHostToDevice);

  cudaMalloc(&d2, sizeof(Complex) * newsize * newsize * newsize);

  cudaMemcpy(d2, h2, sizeof(Complex) * newsize * newsize * newsize, cudaMemcpyHostToDevice);

  // Kernel Block and Grid Size.
  const dim3 blockSize(16, 16, 1);
  const dim3 gridSize((newsize) / 16 + 1, (newsize) / 16 + 1, 1);


  /*signalFFT3D(d1, newsize, newsize, newsize);
  signalFFT3D(d2, newsize, newsize, newsize);

  signalIFFT3D(d1, newsize, newsize, newsize);
  signalIFFT3D(d2, newsize, newsize, newsize);

  cudaMemcpy(h1, d1, sizeof(Complex) * (newsize + newsize + newsize), cudaMemcpyDeviceToHost);
  cudaMemcpy(h2, d2, sizeof(Complex) * (newsize + newsize + newsize), cudaMemcpyDeviceToHost);

  normData(h1, (newsize + newsize + newsize), newsize * newsize * newsize);
  normData(h2, (newsize + newsize + newsize), newsize * newsize * newsize);

  printHostData(h1, s1 + s1 + s1, "h1 after");
  printHostData(h2, s1 + s1 + s1, "h2 after");
  printf ("\n");*/


  signalFFT3D(d1, newsize, newsize, newsize);
  signalFFT3D(d2, newsize, newsize, newsize);

  cudaMemcpy(h1, d1, sizeof(Complex) * (newsize + newsize + newsize), cudaMemcpyDeviceToHost);
  cudaMemcpy(h2, d2, sizeof(Complex) * (newsize + newsize + newsize), cudaMemcpyDeviceToHost);
  printf ("On CPU\n");
  for (i = 0; i < newsize; i++) {
    temp.x = (h1[i].x * h2[i].x) - (h1[i].y * h2[i].y);
    temp.y = (h1[i].x * h2[i].y) + (h1[i].x * h2[i].y);
    printf ("      %f %f      %f %f      %f %f\n", h1[i].x, h1[i].y, h2[i].x, h2[i].y, temp.x, temp.y);
  }
  printf("\n\n");

  //printf("on gpu\n");
  pwProd<<<gridSize, blockSize>>>(d1, newsize, d2, newsize);
  //printf("\n\n");

  signalIFFT3D(d1, newsize, newsize, newsize);
  //signalIFFT3D(d2, newsize, newsize, newsize);

  cudaMemcpy(h1, d1, sizeof(Complex) * (newsize + newsize + newsize), cudaMemcpyDeviceToHost);
  //cudaMemcpy(h2, d2, sizeof(Complex) * (newsize + newsize + newsize), cudaMemcpyDeviceToHost);*/
  printHostData(h1, s1 + s1 + s1, "h1 after");

  printf("\n\nFUNCTION\n\n");

  cudaConvolution3D(d1, newsize, d2, newsize, blockSize, gridSize);

  //normData(h1, (newsize + newsize + newsize), newsize * newsize * newsize);

  printHostData(h1, s1 + s1 + s1, "h1 after");
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

  //Odeenfft();

  cudaDeviceReset();

  trittfft();

  return 0;
}
