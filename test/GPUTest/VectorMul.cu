/* MatrixMul.cu:
*
*/

#include <stdio.h>
#include <cuda.h> // CUDA Include.

// Stuff that executes on the CUDA device
__global__ void square_array(float *a, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
      a[idx] = a[idx] * a[idx];
}

// main routine that executes on the host
int main(void)
{

  // Data pointers for the CPU and the GPU.
  float *a_h, *a_d;

  // Array size.
  const int N = 10;
  size_t size = N * sizeof(float);

  // Allocate Memory on host and the device.
  a_h = (float *) malloc(size);
  cudaMalloc((void **) &a_d, size);

  // Decide block size. A block size determines
  // the amount of data per computation.
  int block_size = 4;
  int n_blocks = N/block_size + (N % block_size  == 0 ? 0:1);

  // Initialize host array and copy to device.
  for (int i = 0; i < N; i++)
    a_h[i] = (float) i;
  cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);

  // Caclulate stuff on the decvice.
  square_array <<< n_blocks, block_size >>> (a_d, N);

  // Copy results, back, from the GPU to the CPU.
  cudaMemcpy(a_h, a_d, sizeof(float)*N, cudaMemcpyDeviceToHost);

  // Print
  for (int i=0; i<N; i++)
    printf("%d %f\n", i, a_h[i]);

  // Cleanup
  free(a_h);
  cudaFree(a_d);

}
