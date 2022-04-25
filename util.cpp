#include "util.h"
#include <cstdlib>
#include <stdio.h>
#include <ctime>
#include <iostream>

void parseArgs(int argc, char **argv, int& arrSize) {
    arrSize = 1000; // default size
    if(argc > 1) {
        arrSize = atoi(argv[1]);
    }
}

void fillArrayWithNumbers(int *arr, int arrSize) {
    // TODO: Rename to initialize Ranodm array
    (time(NULL));
    for (int i = 0; i < arrSize; i++) {
        arr[i] = rand() % 100000; // TODO: INT_MAX;
    }
}

void printArray(int *arr, int arrSize) {
    for (int i=0; i < arrSize; i++)
        printf("%d ", arr[i]);
    printf("\n");
}
