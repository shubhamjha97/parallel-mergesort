#include <cstdio>
#include <omp.h>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <algorithm>

#define DBG 1
#define SHOWTIME 1

#define NUMBERS_BIG 100000000 //2000000
#define NUMBERS_DBG 10

#define MAX_BIG 100000 //1000000
#define MAX_DBG 50

#define NUM_THREADS 4
#define NUMBERS ((DBG == 1) ? NUMBERS_DBG : NUMBERS_BIG)
#define MAX_NUMBER ((DBG == 1) ? MAX_DBG : MAX_BIG)

void inlinePrintArray(int *A, int from, int to) {
    for (int i = from; i < to; i++) {
        printf("%d, ", A[i]);
    }
}

void printArray(int* A, int size) {
    printf("\n");
    for (int i = 0; i < size; i++) {
        printf("%d, ", A[i]);
    }
    printf("\n");
}

void printTime(time_t t1, time_t t2, const char* solutionType) {
    printf("\nTime in seconds (mergesort %s): %f", solutionType, difftime(t2, t1));
}

/* debug */
// TODO: remove
void debugPrintMergeSort(int* arr, int left_start, int right_start, char c) {
    if (DBG) {
        printf("\n%c <%d, %d> ", c, left_start, right_start);
        inlinePrintArray(arr, left_start, right_start);
        fflush(stdout);
    }
}

// TODO: Remove
void debugPrintParralelMergeSort(int left_start, int right_start, int num, int threads) {
    if (DBG) {
        printf("\nthread=%d, num_threads=%d <%d, %d>", num, threads, left_start, right_start);
        fflush(stdout);
    }
}

void checkIfCorrectlySorted(int* arr) {
    bool correct = true;
    for (int i = 0; i < NUMBERS - 1; i++) {
        if (arr[i] > arr[i + 1]) {
            printf("\n\n-----------ERROR!-----------\n\n");
            correct = false;
            break;
        }
    }
    if (correct) {
        printf("\n----------- OK ------------");
    }
}

/* merge sort */
void fillArrayWithNumbers(int* numbers) {
    int i, from, to;
    srand(time(NULL));
    for (i = 0; i < NUMBERS; i++) {
        numbers[i] = rand() % MAX_NUMBER;
    }

    if (DBG) {
        printArray(numbers, NUMBERS);
    }
}

/* Function to merge the two haves arr[l..m] and arr[m+1..r] of array arr[] */
void merge(int arr[], int left_start, int mid, int right_start, int* tmp) {
//    if (DBG) {
//        printf("\nmerge: [ ");
//        inlinePrintArray(arr, left_start, mid);
//        printf(" ] with [ ");
//        inlinePrintArray(arr, mid, right_start);
//        printf(" ]");
//        fflush(stdout);
//    }

    int i, j, k;
    int left_half_size = mid - left_start + 1;
    int right_half_size = right_start - mid;

    /* create temp arrays */
    int* L = &tmp[left_start];
    int* R = &tmp[mid + 1];

    /* Copy data to temp arrays L[] and R[] */
    for (i = 0; i < left_half_size; i++) {
        L[i] = arr[left_start + i];
    }
    for (j = 0; j < right_half_size; j++) {
        R[j] = arr[mid + 1 + j];
    }

    /* Merge the temp arrays back into arr[l..r]*/
    i = 0;
    j = 0;
    k = left_start;
    while (i < left_half_size && j < right_half_size) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        }
        else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    /* Copy the remaining elements of L[], if there are any */
    while (i < left_half_size) {
        arr[k] =L[i];
        i++;
        k++;
    }

    /* Copy the remaining elements of R[], if there are any */
    while (j < right_half_size) {
        arr[k] = R[j];
        j++;
        k++;
    }
}

void mergeSort(int* arr, int left_start, int right_start, int* tmp) {
    if (left_start < right_start)
    {
        // Same as (l+r)/2, but avoids overflow for large l and h
        int m = left_start + (right_start - left_start) / 2;

        // Sort first and second halves
//        debugPrintMergeSort(arr, left_start, m, 'l');
        mergeSort(arr, left_start, m, tmp);

//        debugPrintMergeSort(arr, m + 1, right_start, 'r');
        mergeSort(arr, m + 1, right_start, tmp);

        merge(arr, left_start, m, right_start, tmp);
//        debugPrintMergeSort(arr, left_start, right_start, 'm');
    }
}

void mergeSortParallel(int* arr, int left_start, int right_start, int* tmp) {
    if (left_start < right_start) {
        if ((right_start - left_start) > (DBG ? 1 : 10000)) {
            int num = omp_get_thread_num();
            int threads = omp_get_num_threads();

            int m = left_start + (right_start - left_start) / 2;

#pragma omp task
            {
//                debugPrintParralelMergeSort(left_start, m, num, threads); // TODO: remove
                mergeSortParallel(arr, left_start, m, tmp);
            }

#pragma omp task
            {
//                debugPrintParralelMergeSort(m + 1, right_start, num, threads); // TODO: remove
                mergeSortParallel(arr, m + 1, right_start, tmp);
            }

#pragma omp taskwait

            merge(arr, left_start, m, right_start, tmp);
        }
        else {
            mergeSort(arr, left_start, right_start, tmp);
        }
    }
}

int main() {
    omp_set_nested(1);

    const int numbersSize = NUMBERS * sizeof(int);
    time_t time1, time2;
    int* tmp = (int*)malloc(numbersSize);
    int* numbersPar = (int*)malloc(numbersSize);
    fillArrayWithNumbers(numbersPar);


    #pragma omp target data map(tofrom:tmp[0:NUMBERS]) map(tofrom:numbersPar[0:NUMBERS])
    {

        // TODO: Remove
        int num_devices = omp_get_num_devices();
        printf("Number of available devices %d\n", num_devices);

        #pragma omp target
        {
            if (omp_is_initial_device()) { // TODO: remove
                printf("Running on host\n");
            } else {
                printf("Running on device\n");
            }

            #pragma omp parallel
            {
                mergeSortParallel(numbersPar, 0, NUMBERS - 1, tmp);
            }
        }
    }

    inlinePrintArray(numbersPar, 0, NUMBERS);
    printf("\n");
    checkIfCorrectlySorted(numbersPar);
    printf("\n");

    return 0;
}
