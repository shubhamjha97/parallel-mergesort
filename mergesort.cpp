#include<stdlib.h>
#include "util.h"

using namespace std;

void merge(int arr[], int l, int m, int r) {
    int lSize = m - l + 1, rSize = r - m;
    int left[lSize], right[rSize];

    // Copy elements to temporary array
    for (int i = 0; i < lSize; i++)
        left[i] = arr[l + i];
    for (int i = 0; i < rSize; i++)
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

void mergeSort(int* arr, int l, int r) {
    if (l < r) {
        int m = l + ( r - l ) / 2;
        mergeSort(arr, l, m);
        mergeSort(arr, m+1, r);
        merge(arr, l, m, r);
    }
}

int main(int argc, char **argv) {
    int arrSize;
    parseArgs(argc, argv, arrSize);

    // initialize array
    int arraySizeBytes = arrSize * sizeof(int);
    int *arr = (int*) malloc(arraySizeBytes);
    initializeRandomArray(arr, arrSize);

    // sort
    mergeSort(arr, 0, arrSize - 1);

//    printArray(arr, arrSize);
    return 0;
}
