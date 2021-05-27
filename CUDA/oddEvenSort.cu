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


__global__ void oddEvenSortKernel(float* a, bool* stopFlag, int N, bool evenArr, bool iterationEven) {
	int index = blockIdx.x * BLOCK_SIZE + threadIdx.x; // global thread x

	int i = index * 2;

	if (iterationEven)
	{
		//swap
		if (a[i] > a[i + 1])
		{
			auto tmp = a[i];
			a[i] = a[i + 1];
			a[i + 1] = tmp;
			stopFlag[0] = false;
		}
	}
	else
	{
		if (!(evenArr && index == N / 2 - 1))
		{
			i++;
			//swap
			if (a[i] > a[i + 1])
			{
				auto tmp = a[i];
				a[i] = a[i + 1];
				a[i + 1] = tmp;
				stopFlag[0] = false;
			}
			i--;
		}
	}
}

void printArr(vector<float> a) {
	int N = a.size();
	for (int i = 0; i < N; i++) {
		cout << a[i] << " ";
	}
	cout << endl;
}

//CPU side
vector<float> runOddEvenSort(int N, vector<float> h_a) {
	//Matrix size in bytes
	size_t byteSize = N * sizeof(float);

	//Allocate device memory
	float* d_a;
	bool* d_stopFlag;
	cudaMalloc(&d_a, byteSize);
	cudaMalloc(&d_stopFlag, sizeof(bool));

	//Copy data to device
	cudaMemcpy(d_a, h_a.data(), byteSize, cudaMemcpyHostToDevice);

	//Blocks per grid dimension
	int BlkGrdDim = (int)ceil((float)N / BLOCK_SIZE);
	dim3 threads(BLOCK_SIZE, 1);
	dim3 blocks(BlkGrdDim, 1);
	
	//Run kernel
	bool evenArr = (N % 2) == 0 ? true : false;
	bool stopFlag = false;
	bool iterationEven = true;

	while (true)
	{	
		if (stopFlag)
			break;

		cudaMemcpy(d_stopFlag, new bool(true), sizeof(bool), cudaMemcpyHostToDevice);
		oddEvenSortKernel <<<blocks, threads>>> (d_a, d_stopFlag, N, evenArr, iterationEven);
		cudaThreadSynchronize();
		cudaMemcpy(&stopFlag, d_stopFlag, sizeof(bool), cudaMemcpyDeviceToHost);

		iterationEven = !iterationEven;
	}

	//Copy back to host
	cudaMemcpy(h_a.data(), d_a, byteSize, cudaMemcpyDeviceToHost);

	//Free memory on device
	cudaFree(d_a);
	cudaFree(d_stopFlag);

	return h_a;
}

//CPU side
int main() {
	////Matrix size N x N
	//const int N = 2560;
	//const int N = 5120;
	const int N = 10240;
	//const int N = 15360;
	//const int N = 20480;
	//const int N = 30720;
	//const int N = 40960;

	vector<float> time_list;
	int launchIter = 6;
	int warmupLaunches = 1;

	//Timer stuff
	float time;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//Matrices
	vector<float> h_a(N);

	for (int i = 0; i < launchIter; i++) {
		//Initialize matrices
		generate(h_a.begin(), h_a.end(), []() { return rand() % 1000; });

		//Start timer here
		cudaEventRecord(start, 0);

		runOddEvenSort(N, h_a);

		//Stop timer here
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time, start, stop);
		cout << "True time = " << time << endl;
		time_list.push_back(time);
	}

	for (int i = 0; i < warmupLaunches; i++)
		time_list.erase(time_list.begin());

	float sumTime = 0;
	for (auto el : time_list)
	{
		sumTime += el;
	}

	cout << endl << "Avg time = " << round(sumTime / (launchIter - warmupLaunches)) << endl;

	//Event variables destruction (lol)
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cout << "Done" << endl;
	return 0;
}
