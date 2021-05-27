//#include <algorithm>
//#include <cassert>
//#include <cstdlib>
//#include <functional>
//#include <iostream>
//#include <vector>
//#include <cuda_runtime.h>
//#include "device_launch_parameters.h"
//#include <random>
//
//using std::cout;
//using std::generate;
//using std::vector;
//
//using namespace std;
//
//#define BLOCK_SIZE 32
//
//
//__global__ void oddEvenSort(float* a, bool* stopFlag, int N, bool evenArr) {
//	int index = blockIdx.x * BLOCK_SIZE + threadIdx.x; // global thread x
//
//	int i = index * 2;
//	bool iterationEven = true;
//
//	while (true)
//	{
//		if (stopFlag[0])
//			break;
//		__syncthreads();
//		if (index == 0)
//			stopFlag[0] = true;
//		__syncthreads();
//
//		if (iterationEven)
//		{
//			//swap
//			if (a[i] > a[i + 1])
//			{
//				auto tmp = a[i];
//				a[i] = a[i + 1];
//				a[i + 1] = tmp;
//				stopFlag[0] = false;
//			}
//		}
//		else
//		{
//			if (!(evenArr && index == N / 2 - 1))
//			{
//				i++;
//				//swap
//				if (a[i] > a[i + 1])
//				{
//					auto tmp = a[i];
//					a[i] = a[i + 1];
//					a[i + 1] = tmp;
//					stopFlag[0] = false;
//				}
//				i--;
//			}
//		}
//		iterationEven = !iterationEven;
//	}
//}
//
//void printArr(vector<float> a) {
//	int N = a.size();
//	for (int i = 0; i < N; i++) {
//		cout << a[i] << " ";
//	}
//	cout << endl;
//}
//
////CPU side
//int main() {
//	////Matrix size N x N
//	const int N = 8192;
//	//const int N = 512;
//	//const int N = 1024;
//	//const int N = 1536;
//	//const int N = 2048;
//	//const int N = 3072;
//	//const int N = 4096 * 4096;
//	//const int N = 10;
//
//	//Timer stuff
//	float time;
//	cudaEvent_t start, stop;
//	cudaEventCreate(&start);
//	cudaEventCreate(&stop);
//
//	//Matrix size in bytes
//	size_t byteSize = N * sizeof(float);
//
//	//Matrices
//	vector<float> h_a(N);
//
//	//Initialize matrices
//	generate(h_a.begin(), h_a.end(), []() { return rand() % 1000; });
//
//	//printArr(h_a);
//
//	//Start timer here
//	cudaEventRecord(start, 0);
//
//	//Allocate device memory
//	float* d_a;
//	bool* d_stopFlag;
//	cudaMalloc(&d_a, byteSize);
//	cudaMalloc(&d_stopFlag, sizeof(bool));
//
//	//Copy data to device
//	cudaMemcpy(d_a, h_a.data(), byteSize, cudaMemcpyHostToDevice);
//	cudaMemcpy(d_stopFlag, false, sizeof(bool), cudaMemcpyHostToDevice);
//
//	//Blocks per grid dimension
//	int BlkGrdDim = (int)ceil((float)N / 2 / BLOCK_SIZE);
//	dim3 threads(BLOCK_SIZE, 1);
//	dim3 blocks(BlkGrdDim, 1);
//	
//	//Run kernel
//	bool evenArr = (N % 2) == 0 ? true : false;
//	oddEvenSort <<<blocks, threads>>> (d_a, d_stopFlag, N, evenArr);
//	cudaThreadSynchronize();
//
//	//Copy back to host
//	cudaMemcpy(h_a.data(), d_a, byteSize, cudaMemcpyDeviceToHost);
//
//	//printArr(h_a);
//
//	//Free memory on device
//	cudaFree(d_a);
//	cudaFree(d_stopFlag);
//
//	//Stop timer here
//	cudaEventRecord(stop, 0);
//	cudaEventSynchronize(stop);
//	cudaEventElapsedTime(&time, start, stop);
//	cout << "Time = " << time << endl << endl;	
//
//	//Event variables destruction (lol)
//	cudaEventDestroy(start);
//	cudaEventDestroy(stop);
//
//	cout << "Done" << endl;
//
//	return 0;
//}
