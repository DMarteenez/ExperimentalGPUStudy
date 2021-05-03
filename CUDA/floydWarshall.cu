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


__global__ void floydWarshall(int k, float* d_graphMinDist, float* d_graphPath, int N) {

	int col = blockIdx.x * blockDim.x + threadIdx.x; //Each thread along x is assigned to a matrix column
	int row = blockIdx.y; //Each block along y is assigned to a matrix row

	if (col >= N) return;

	int arrayIndex = N * row + col;

	//All the blocks load the entire k-th column into shared memory
	__shared__ float d_graphMinDist_row_k;
	if (threadIdx.x == 0) d_graphMinDist_row_k = d_graphMinDist[N * row + k];
	__syncthreads();

	if (d_graphMinDist_row_k == DBL_MAX)   //If element (row, k) = infinity, no update is needed
		return;

	float d_graphMinimumDistances_k_col = d_graphMinDist[k * N + col];
	if (d_graphMinimumDistances_k_col == DBL_MAX)    //If element (k, col) = infinity, no update is needed
		return;

	float candidateBetterDist = d_graphMinDist_row_k + d_graphMinimumDistances_k_col;
	if (candidateBetterDist < d_graphMinDist[arrayIndex]) {
		d_graphMinDist[arrayIndex] = candidateBetterDist;
		d_graphPath[arrayIndex] = d_graphPath[k * N + col];
	}
}


int main() {
	//Matrix size N x N
	int N = 1024;

	//Timer stuff
	float time;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//Matrix size in bytes
	size_t byteSize = N * N * sizeof(float);

	//Matrices
	vector<float> h_graphMinDist(N * N);
	vector<float> h_graphPath(N * N);

	//Initialize matrices
	generate(h_graphMinDist.begin(), h_graphMinDist.end(), []() { return rand() % 100; });

	//Allocate device memory (device = GPU)
	float* d_graphMinDist;
	float* d_graphPath;
	cudaMalloc(&d_graphMinDist, byteSize);
	cudaMalloc(&d_graphPath, byteSize);

	//Copy data to device
	cudaMemcpy(d_graphMinDist, h_graphMinDist.data(), byteSize, cudaMemcpyHostToDevice);

	//Blocks per grid dimension
	int BlkGrdDim = (int)ceil(N / BLOCK_SIZE);

	//dim3 - cuda int vector https://codeyarns.com/tech/2011-02-16-cuda-dim3.html
	dim3 blocks(BlkGrdDim, N);

	cout << "Starting" << endl;

	//Start timer
	cudaEventRecord(start, 0);

	//Run kernel iterations
	for (int k = 0; k < N; k++) {
		floydWarshall <<<blocks, BLOCK_SIZE >> > (k, d_graphMinDist, d_graphPath, N);
	}

	cudaThreadSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	cout << time << endl;

	//Copy back to host
	cudaMemcpy(h_graphMinDist.data(), d_graphMinDist, byteSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_graphPath.data(), d_graphPath, byteSize, cudaMemcpyDeviceToHost);

	//Free memory on device
	cudaFree(d_graphMinDist);
	cudaFree(d_graphPath);

	//Event variables destruction (lol)
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cout << "Done" << endl;

	return 0;
}