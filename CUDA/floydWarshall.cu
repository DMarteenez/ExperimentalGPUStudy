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


__global__ void floydWarshall(int k, float* d_graphMinDist, int N) {

	int col = blockIdx.x * blockDim.x + threadIdx.x; //Each thread along x is assigned to a matrix column
	int row = blockIdx.y; //Each block along y is assigned to a matrix row

	if (col >= N) return;

	int arrayIndex = N * row + col;

	float candidateBetterDist = d_graphMinDist[N * row + k] + d_graphMinDist[k * N + col];

	if (candidateBetterDist < d_graphMinDist[arrayIndex])
		d_graphMinDist[arrayIndex] = candidateBetterDist;
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
	vector<float> h_graphMinDist(N * N);

	//Initialize matrices
	generate(h_graphMinDist.begin(), h_graphMinDist.end(), []() { return rand() % 100; });

	//Start timer
	cudaEventRecord(start, 0);

	//Allocate device memory (device = GPU)
	float* d_graphMinDist;
	cudaMalloc(&d_graphMinDist, byteSize);

	//Copy data to device
	cudaMemcpy(d_graphMinDist, h_graphMinDist.data(), byteSize, cudaMemcpyHostToDevice);

	//Blocks per grid dimension
	int BlkGrdDim = (int)ceil((float)N / BLOCK_SIZE);

	//dim3 - cuda int vector https://codeyarns.com/tech/2011-02-16-cuda-dim3.html
	dim3 blocks(BlkGrdDim, N);

	cout << "Starting" << endl;

	//Run kernel iterations
	for (int k = 0; k < N; k++) {
		floydWarshall <<<blocks, BLOCK_SIZE>>> (k, d_graphMinDist, N);
		cudaThreadSynchronize();
	}

	//Copy back to host
	cudaMemcpy(h_graphMinDist.data(), d_graphMinDist, byteSize, cudaMemcpyDeviceToHost);

	//printMatrix(h_graphMinDist, N);

	//Free memory on device
	cudaFree(d_graphMinDist);
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	cout << time << endl;

	//Event variables destruction (lol)
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cout << "Done" << endl;

	return 0;
}