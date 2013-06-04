#include <cudaConv.h>

int main()
{

  Complex *h1, *h2, *d1, *d2;

  int s1, s2, newsize, i, dim;


  if (checkCUDA()) {
    printf ("CUDA FAIL\n");
    exit(1);
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

      cudaMalloc(&d1, sizeof(Complex) * newsize);
      cudaMalloc(&d2, sizeof(Complex) * newsize);

      cudaMemcpy(d1, h1, sizeof(Complex) * newsize, cudaMemcpyHostToDevice);
      cudaMemcpy(d2, h2, sizeof(Complex) * newsize, cudaMemcpyHostToDevice);

      cudaConvolution(d1, newsize, d2, newsize, blockSize, gridSize);

      cudaMemcpy(h1, d1, sizeof(Complex) * newsize, cudaMemcpyDeviceToHost);

      normData(h1, newsize, newsize);

      printHostData(h1, newsize, "Conv");

      free(h1); free(h2);
      cudaFree(d1); cudaFree(d2);

      cudaDeviceReset();
  }

  return 0;
}
