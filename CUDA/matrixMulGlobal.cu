#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <random>

using std::cout;
using std::generate;
using std::vector;

using namespace std;

#define BLOCK_SIZE 32


__global__ void matrixMul(float* a, float* b, float* c, int N) {
	int gx = blockIdx.x * BLOCK_SIZE + threadIdx.x; // global thread x
	int gy = blockIdx.y * BLOCK_SIZE + threadIdx.y; // global thread y

	float sum = 0.f;

	for (int r = 0; r < N; r++)
	{
		sum += a[gy * N + r] * b[gx + r * N];
	}

	c[gy * N + gx] = sum;
}

void printMatrix(vector<float> a, int N) {
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			cout << a[i * N + j] << " ";
		}
		cout << endl;
	}
	cout << endl;
}

//CPU side
int main() {
	////Matrix size N x N
	//const int N = 256;
	//const int N = 512;
	//const int N = 1024;
	//const int N = 1536;
	//const int N = 2048;
	//const int N = 3072;
	const int N = 4096;

	//Timer stuff
	float time;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//Matrix size in bytes
	size_t byteSize = N * N * sizeof(float);

	//Matrices
	vector<float> h_a(N * N);
	vector<float> h_b(N * N);
	vector<float> h_c(N * N);

	//Initialize matrices
	generate(h_a.begin(), h_a.end(), []() { return rand() % 100; });
	generate(h_b.begin(), h_b.end(), []() { return rand() % 100; });

	//printMatrix(h_a, N);
	//printMatrix(h_b, N);

	//Start timer here
	cudaEventRecord(start, 0);

	//Allocate device memory (device = GPU)
	float* d_a, * d_b, * d_c;
	cudaMalloc(&d_a, byteSize);
	cudaMalloc(&d_b, byteSize);
	cudaMalloc(&d_c, byteSize);

	//Copy data to device
	cudaMemcpy(d_a, h_a.data(), byteSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b.data(), byteSize, cudaMemcpyHostToDevice);

	//Blocks per grid dimension
	int BlkGrdDim = (int)ceil((float)N / BLOCK_SIZE);

	//dim3 - cuda int vector https://codeyarns.com/tech/2011-02-16-cuda-dim3.html
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocks(BlkGrdDim, BlkGrdDim);

	//Run kernel
	matrixMul <<<blocks, threads>>> (d_a, d_b, d_c, N);
	cudaThreadSynchronize();

	//Copy back to host
	cudaMemcpy(h_c.data(), d_c, byteSize, cudaMemcpyDeviceToHost);

	//printMatrix(h_c, N);

	//Free memory on device
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	//Stop timer here
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	cout << "Time = " << time << endl << endl;
	

	//Event variables destruction (lol)
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cout << "Done" << endl;

	return 0;
}
