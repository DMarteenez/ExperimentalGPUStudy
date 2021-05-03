//#include <algorithm>
//#include <cassert>
//#include <cstdlib>
//#include <functional>
//#include <iostream>
//#include <vector>
//#include <cuda_runtime.h>
//#include "device_launch_parameters.h"
//
//using std::cout;
//using std::generate;
//using std::vector;
//
//using namespace std;
//
//#define BLOCK_SIZE 32
//
////GPU side
////__global__ void matrixMul(float* a, float* b, float* c, float N) {
////	
////	int bx = blockIdx.x; // block number by x
////	int by = blockIdx.y; // block number by y
////	int tx = threadIdx.x; // thread number in block by x
////	int ty = threadIdx.y; // thread number in block by y
////	
////	int row = N * (BLOCK_SIZE * by + ty); //row from a
////	int col = BLOCK_SIZE * bx + tx; //col from b
////	int ic = row + col; //element num in c
////	float sum = 0.f;
////
////	//Calculate current row and column into corresponding cell of matrix c
////	for (int k = 0; k < N; k++) {
////		// Accumulate results for a single element
////		sum += a[row + k] * b[k * N + col];
////	}
////	c[ic] = sum;
////}
//
////Copied from shared version
//__global__ void matrixMul(float* a, float* b, float* c, int N) {
//
//	int gx = blockIdx.x * BLOCK_SIZE + threadIdx.x; // block number by x
//	int gy = blockIdx.y * BLOCK_SIZE + threadIdx.y; // block number by y
//	int tx = threadIdx.x; // thread number in block by x
//	int ty = threadIdx.y; // thread number in block by y
//
//	float sum = 0.f;
//
//	for (int i = 0; i < N; i += BLOCK_SIZE)
//	{
//		for (int k = 0; k < BLOCK_SIZE; k++)
//			sum += a[gy * N + k + i] * b[(k + i) * N + gx];
//	}
//	c[gy * N + gx] = sum;
//}
//
////CPU side
//int main() {
//	//Matrix size N x N
//	int N = 2048;
//
//	//Timer stuff
//	float time;
//	cudaEvent_t start, stop;
//	cudaEventCreate(&start);
//	cudaEventCreate(&stop);
//
//	//Matrix size in bytes
//	size_t byteSize = N * N * sizeof(float);
//
//	//Matrices
//	vector<float> h_a(N * N);
//	vector<float> h_b(N * N);
//	vector<float> h_c(N * N);
//
//	//Initialize matrices
//	generate(h_a.begin(), h_a.end(), []() { return rand() % 100; });
//	generate(h_b.begin(), h_b.end(), []() { return rand() % 100; });
//
//	//Allocate device memory (device = GPU)
//	float* d_a, * d_b, * d_c;
//	cudaMalloc(&d_a, byteSize);
//	cudaMalloc(&d_b, byteSize);
//	cudaMalloc(&d_c, byteSize);
//
//	//Copy data to device
//	cudaMemcpy(d_a, h_a.data(), byteSize, cudaMemcpyHostToDevice);
//	cudaMemcpy(d_b, h_b.data(), byteSize, cudaMemcpyHostToDevice);
//
//	//Blocks per grid dimension
//	int BlkGrdDim = (int)ceil(N / BLOCK_SIZE);
//
//	//dim3 - cuda int vector https://codeyarns.com/tech/2011-02-16-cuda-dim3.html
//	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
//	dim3 blocks(BlkGrdDim, BlkGrdDim);
//
//	//Start timer
//	cudaEventRecord(start, 0);
//
//	//Run kernel
//	matrixMul << <blocks, threads >> > (d_a, d_b, d_c, N);
//
//	cudaThreadSynchronize();
//	cudaEventRecord(stop, 0);
//	cudaEventSynchronize(stop);
//	cudaEventElapsedTime(&time, start, stop);
//	cout << time << endl;
//
//	//Copy back to host
//	cudaMemcpy(h_c.data(), d_c, byteSize, cudaMemcpyDeviceToHost);
//
//	//Free memory on device
//	cudaFree(d_a);
//	cudaFree(d_b);
//	cudaFree(d_c);
//
//	//Event variables destruction (lol)
//	cudaEventDestroy(start);
//	cudaEventDestroy(stop);
//
//	cout << "Done" << endl;
//
//	return 0;
//}