omp:
	/ext3/gcc-offload/install/bin/gcc -fopenmp mergesort-openmp.cpp util.cpp -o mergesort_omp -lstdc++

seq:
	gcc mergesort.cpp util.cpp -o mergesort_seq -lstdc++

cuda:
	nvcc mergesort-cuda.cu util.cpp  -o mergesort_cuda

clean:
	rm -rf mergesort_cuda mergesort_omp mergesort_seq
