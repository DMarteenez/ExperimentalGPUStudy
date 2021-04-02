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

//GPU side
//Each cell in output matrix has it's own thread to calculate it
__global__ void matrixMul(const int* a, const int* b, int* c, int N) {
	//Get current thread's row and column
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	//Calculate current row and column into corresponding cell of matrix c
	c[row * N + col] = 0;
	for (int k = 0; k < N; k++) {
		// Accumulate results for a single element
		c[row * N + col] += a[row * N + k] * b[k * N + col];
	}
}

//CPU side
int main() {
	//Matrix size N x N
	int N = 1000;

	//Matrix size in bytes
	size_t byteSize = N * N * sizeof(int);

	//Matrices
	vector<int> h_a(N * N);
	vector<int> h_b(N * N);
	vector<int> h_c(N * N);

	//Initialize matrices
	generate(h_a.begin(), h_a.end(), []() { return rand() % 100; });
	generate(h_b.begin(), h_b.end(), []() { return rand() % 100; });

	//Allocate device memory
	int* d_a, * d_b, * d_c;
	cudaMalloc(&d_a, byteSize);
	cudaMalloc(&d_b, byteSize);
	cudaMalloc(&d_c, byteSize);

	//Copy data to device
	cudaMemcpy(d_a, h_a.data(), byteSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b.data(), byteSize, cudaMemcpyHostToDevice);

	//Threads per CTA dimension, CTA = 16x16
	int ThrCtaDim = 16;

	//Blocks per grid dimension
	int BlkGrdDim = (int)ceil(N / ThrCtaDim);

	//dim3 - cuda int vector https://codeyarns.com/tech/2011-02-16-cuda-dim3.html
	dim3 threads(ThrCtaDim, ThrCtaDim);
	dim3 blocks(BlkGrdDim, BlkGrdDim);

	//Kernel
	matrixMul<<<BlkGrdDim, ThrCtaDim>>>(d_a, d_b, d_c, N);

	//Copy back to host
	cudaMemcpy(h_c.data(), d_c, byteSize, cudaMemcpyDeviceToHost);

	//Free memory on device
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	cout << "Done" << endl;
	
	return 0;
}