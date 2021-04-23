#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

using std::cout;
using std::generate;
using std::vector;

using namespace std;

#define BLOCK_SIZE 32

//GPU side
__global__ void matrixMul(int* a, int* b, int* c, int N) {
	
	int bx = blockIdx.x; // block number by x
	int by = blockIdx.y; // block number by y
	int tx = threadIdx.x; // thread number in block by x
	int ty = threadIdx.y; // thread number in block by y
	
	int aBegin = N * BLOCK_SIZE * by;
	int aEnd = aBegin + N - 1;
	int bBegin = BLOCK_SIZE * bx;
	int aStep = BLOCK_SIZE;
	int bStep = BLOCK_SIZE * N;
	int sum = 0;

	__shared__ int as[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ int bs[BLOCK_SIZE][BLOCK_SIZE];

	for (int ia = aBegin, ib = bBegin; ia <= aEnd; ia += aStep, ib += bStep)
	{
		as[tx][ty] = a[ia + N * ty + tx]; //copy from global mem to shared
		bs[tx][ty] = b[ib + N * ty + tx];
		__syncthreads();
		for (int k = 0; k < BLOCK_SIZE; k++) 
			sum += as[k][ty] * bs[tx][k];
		__syncthreads();
	}
	c[aBegin + bBegin + ty * N + tx] = sum;
}

//CPU side
int main() {
	//Matrix size N x N
	int N = 2048;

	//Timer stuff
	float time;
	cudaEvent_t start, stop;
	cudaEventCreate(&start); 
	cudaEventCreate(&stop);

	//Matrix size in bytes
	size_t byteSize = N * N * sizeof(int);

	//Matrices
	vector<int> h_a(N * N);
	vector<int> h_b(N * N);
	vector<int> h_c(N * N);

	//Initialize matrices
	generate(h_a.begin(), h_a.end(), []() { return rand() % 100; });
	generate(h_b.begin(), h_b.end(), []() { return rand() % 100; });

	//Allocate device memory (device = GPU)
	int* d_a, * d_b, * d_c;
	cudaMalloc(&d_a, byteSize);
	cudaMalloc(&d_b, byteSize);
	cudaMalloc(&d_c, byteSize);

	//Copy data to device
	cudaMemcpy(d_a, h_a.data(), byteSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b.data(), byteSize, cudaMemcpyHostToDevice);

	//Blocks per grid dimension
	int BlkGrdDim = (int)ceil(N / BLOCK_SIZE);

	//dim3 - cuda int vector https://codeyarns.com/tech/2011-02-16-cuda-dim3.html
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocks(BlkGrdDim, BlkGrdDim);

	//Start timer
	cudaEventRecord(start, 0);

	//Run kernel
	matrixMul<<<blocks, threads>>>(d_a, d_b, d_c, N);

	cudaThreadSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	cout << time << endl;

	//Copy back to host
	cudaMemcpy(h_c.data(), d_c, byteSize, cudaMemcpyDeviceToHost);

	//Free memory on device
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	//Event variables destruction (lol)
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cout << "Done" << endl;
	
	return 0;
}