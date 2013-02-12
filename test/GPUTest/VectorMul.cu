/*
* MatrixMul.cu
*
* Adpated from http://llpanorama.wordpress.com/2008/05/21/my-first-cuda-program/
*
*/

#include <stdio.h>
#include <cuda.h>

// Stuff that executes on the CUDA device.
// Each element of the result array is
// computed by a processing element from a
// block. blockIdx is the block ID of the
// thread. blockDim.x is the number of
// threads per block and threadIdx is the
// thread index within the block.
// Effectively, what this does is that once
// the underlying locks are called on the Device
// memory executing the code below, each thread,
// as per the generics passed to it, then determines
// which element to compute. Obviously in this sample,
// thread selection is naive (and probably the easiest,
// most simple way to do this.)
__global__ void square_array(float *a, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
      a[idx] = a[idx] * a[idx];
}

// Stuff that executes on the Host.
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

  // Decide block size. A block represents the number of thread
  // that are used. The Device is composed of multiple
  // blocks arragned as a grid.
  int block_size = 4;
  int n_blocks = N/block_size + (N % block_size  == 0 ? 0:1);

  // Initialize host array and copy to device.
  for (int i = 0; i < N; i++)
    a_h[i] = (float) i;
  cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);

  // Caclulate stuff on the decvice. Arranges the Device
  // as a gird of size n_blocks with each block consisting of
  // block_size number of processing elements. Each processing
  // element runs a thread.
  square_array <<< n_blocks, block_size >>> (a_d, N);

  // Copy results, back, from the GPU to the CPU.
  cudaMemcpy(a_h, a_d, sizeof(float)*N, cudaMemcpyDeviceToHost);

  // Display Output.
  for (int i=0; i<N; i++)
    printf("%d %f\n", i, a_h[i]);

  // Cleanup.
  free(a_h);
  cudaFree(a_d);

}
