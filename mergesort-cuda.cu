#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "util.h"

__host__
__device__
void merge(int *arr, int l, int r, int m, int tmpIndexStart) {
	int lSize = m - l + 1, rSize = r - m;
	int *left = &arr[tmpIndexStart + l];
	int *right = &arr[tmpIndexStart + m + 1];

    // Copy elements to temporary array
    for(int i = 0; i < lSize; i++)
        left[i] = arr[l + i];
	for(int i = 0; i < rSize; i++)
        right[i] = arr[m + 1 + i];

    // Merge temporary arrays into original array
	int i = 0, j = 0, k = l;
	while (i < lSize && j < rSize) {
		if (left[i] <= right[j]) {
            arr[k++] = left[i++];
        } else {
			arr[k++] = right[j++];
		}
	}

    // Fill in the remaining elements
	while (i < lSize) arr[k++] = left[i++];
	while (j < rSize) arr[k++] = right[j++];
}

__host__
__device__
void mergeSort(int *arr, int leftStart, int rightEnd, int minVectorLength, int vectorLength, int tmpIndexStart) {
	if (leftStart < rightEnd && rightEnd - leftStart >= minVectorLength && rightEnd <= vectorLength) {
		int m = leftStart + (rightEnd - leftStart) / 2;
		mergeSort(arr, leftStart, m, minVectorLength, vectorLength, tmpIndexStart);
		mergeSort(arr, m + 1, rightEnd, minVectorLength, vectorLength, tmpIndexStart);
		merge(arr, leftStart, rightEnd, m, tmpIndexStart);
	}
}

__global__
void mergeSortKernel(int *arr, int vectorLengthPerThread, int minVectorLength, int vectorLength, int tmpIndexStart) {
	int threadId = blockDim.x * blockIdx.x + threadIdx.x;
	int l = threadId * vectorLengthPerThread;
	int r = l + vectorLengthPerThread - 1;
	mergeSort(arr, l, r, minVectorLength, vectorLength, tmpIndexStart);
}

int main(int argc, char **argv) {
    int arrSize;
    parseArgs(argc, argv, arrSize);

	int threadsPerBlock = 512;
	int initialVectorLengthPerThread = 2, vectorLengthPerThread = initialVectorLengthPerThread;
	int numBlocks = ceil((double)arrSize / threadsPerBlock);
	int arrSizePerBlock = arrSize / numBlocks;
	int arrSizeBytes = arrSize * sizeof(int) * 2;
	int tmpIndexStart = arrSize;

    // Initialize array
	int *arr = (int*)malloc(arrSizeBytes);
    initializeRandomArray(arr, arrSize);

    // Copy array to device
    int *arr_d = NULL;
	cudaMalloc((void**)&arr_d, arrSizeBytes);
	cudaMemcpy(arr_d, arr, arrSizeBytes, cudaMemcpyHostToDevice);

	while (vectorLengthPerThread <= arrSizePerBlock) {
		mergeSortKernel<<<numBlocks, threadsPerBlock>>>(arr_d, vectorLengthPerThread, vectorLengthPerThread / initialVectorLengthPerThread, arrSize, tmpIndexStart);
        cudaDeviceSynchronize();
        vectorLengthPerThread *= 2;
    }

    // Copy result to host
    cudaMemcpy(arr, arr_d, arrSizeBytes, cudaMemcpyDeviceToHost);

    int l = vectorLengthPerThread/2, r = arrSize - 1;
	mergeSort(arr, l, r, 1, arrSize, tmpIndexStart);
    merge(arr, 0, arrSize - 1, vectorLengthPerThread/2-1, tmpIndexStart);

    // Free device memory
    cudaFree(arr_d);

	return 0;
}

