#ifndef CUDACONV_H
#define CUDACONV_H

#include <cuda_runtime.h>
#include <cufft.h>
#include <cuda.h>

typedef enum signaltype {REAL, COMPLEX} signal;

typedef float2 Complex;


int
checkCUDA();

void
printHostData(Complex *a, int size, char *msg);


void
printDeviceData(Complex *a, int size, char *msg);

void
normData(Complex *a, int size, float norm);


void
randomFill(Complex *h_signal, int size, int flag);

void
zeroFill(Complex *h_signal, int size);

void
signalFFT(Complex *d_signal, int signal_size);


void
signalIFFT(Complex *d_signal, int signal_size);


__global__ void
pwProd(Complex *signal1, int size1, Complex *signal2, int size2);


void
cudaConvolution(Complex *d_signal1, int size1, Complex *d_signal2,
                int size2, dim3 blockSize, dim3 gridSize);

int
allocateAndPad(Complex **a, int s1, Complex **b, int s2);

#endif
