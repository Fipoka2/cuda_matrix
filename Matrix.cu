#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <random>
#include "Matrix.h"


namespace {
    const unsigned int MAX_THREADS = 1024;
    const unsigned int MAX_BLOCKS = 2147483647;
};

double omp_get_wtime() {
    return 1.0;
}

std::random_device rd;
std::mt19937 mt(rd());
std::uniform_int_distribution<int> dist(1, 1000);

Matrix getRandArray(size_t str, size_t col) {
    int *arr = new int[col * str];
    for (int i = 0; i < col * str; i++) {
        arr[i] = dist(mt);
    }

    Matrix m;
    m.data = arr;
    m.strings = str;
    m.columns = col;

    return m;
}

Matrix getRandArray(size_t size) {
    int *arr = new int[size * size];
#pragma
    for (long i = 0; i < size * size; i++) {
        arr[i] = dist(mt);
    }

    Matrix m;
    m.data = arr;
    m.strings = size;
    m.columns = size;

    return m;
}

Matrix createMatrix(size_t str, size_t col) {
    int *arr = new int[col * str];

    Matrix m;
    m.data = arr;
    m.strings = str;
    m.columns = col;

    return m;
}





void printMatrix(Matrix m) {
    for (int i = 0; i < m.strings; i++) {
        cout << endl;
        for (int j = 0; j < m.columns; j++)
            std::cout << m.data[i * m.strings + j] << ' ';
    }
    cout << endl << endl;

}

__global__ void
sharedCalculate(int *dev_a, int *dev_b, int *dev_c, int *dev_res, unsigned int total) {

    __shared__ int temp[1024];

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i <= total) {
        temp[threadIdx.x] = dev_a[i] > dev_b[i]
                     ? (dev_a[i] > dev_c[i] ? dev_a[i] : dev_c[i])
                     : (dev_b[i] > dev_c[i] ? dev_b[i] : dev_c[i]);
    }
    __syncthreads();
    dev_res[i] = temp[threadIdx.x];
}

__global__ void
globalCalculate(int *dev_a, int *dev_b, int *dev_c, int *dev_res, unsigned int total) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i <= total) {
        dev_res[i] = dev_a[i] > dev_b[i]
                     ? (dev_a[i] > dev_c[i] ? dev_a[i] : dev_c[i])
                     : (dev_b[i] > dev_c[i] ? dev_b[i] : dev_c[i]);
    }
}

MatrixResult cudaCalc(Matrix& a, Matrix& b, Matrix& c, Matrix &result, memoryType type) {

    //time part
    cudaEvent_t copyToDevice, copyToHost, start, stop;
    gpuTime gTime;
    cudaEventCreate ( &start );
    cudaEventCreate ( &copyToDevice );
    cudaEventCreate ( &copyToHost );
    cudaEventCreate ( &stop );
    cudaEventRecord ( copyToDevice, 0 );

    int *dev_a = nullptr;
    int *dev_b = nullptr;
    int *dev_c = nullptr;
    int *dev_res = nullptr;

    cudaSetDevice(0);
    cudaMalloc((void **) &dev_a, a.strings * a.columns * sizeof(int));
    cudaMalloc((void **) &dev_b, b.strings * b.columns * sizeof(int));
    cudaMalloc((void **) &dev_c, c.strings * c.columns * sizeof(int));
    cudaMalloc((void **) &dev_res, result.strings * result.columns * sizeof(int));

    cudaMemcpy(dev_a, a.data, a.strings * a.columns * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b.data, b.strings * b.columns * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c, c.data, c.strings * c.columns * sizeof(int), cudaMemcpyHostToDevice);

    const unsigned int total = c.strings * c.columns;
    const unsigned int numBlocks = floor(double(total / MAX_THREADS)) + 1;

    cudaEventRecord ( start, 0 );

    // Kernel invocation
    switch(type) {
        case global: {
            globalCalculate <<< numBlocks, MAX_THREADS >>> (dev_a, dev_b, dev_c, dev_res, total);
            break;
        }
        case shared: {
            sharedCalculate <<< numBlocks, MAX_THREADS >>> (dev_a, dev_b, dev_c, dev_res, total);
            break;
        }
        default:
            break;
    }

    cudaEventRecord ( stop, 0 );
    cudaEventSynchronize ( stop );

    cudaDeviceSynchronize();
    cudaMemcpy(result.data, dev_res, result.strings * result.columns * sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_res);

    cudaEventRecord ( copyToHost, 0 );
    cudaEventSynchronize ( copyToHost );
    cudaEventElapsedTime ( &(gTime.kernel), start, stop );
    cudaEventElapsedTime ( &(gTime.toDevice), copyToDevice, start );
    cudaEventElapsedTime ( &(gTime.toHost), stop, copyToHost );
    cudaEventElapsedTime ( &(gTime.total), copyToDevice, copyToHost );
    MatrixResult m = {result, gTime};
    return m;
}


