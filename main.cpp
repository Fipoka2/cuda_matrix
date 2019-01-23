#include <iostream>
#include <stdio.h>
#include <string>
#include <cmath>
#include "Matrix.h"
#include <limits>


struct m_time {
    double min = std::numeric_limits<double>::max();
    double average = 0;
    double max = 0;
};

void print(Matrix a, Matrix b, Matrix c, Matrix res, Matrix paral) {
    std::cout << "a"<< endl;
    printMatrix(a);
    std::cout << "b"<< endl;
    printMatrix(b);
    std::cout << "c"<< endl;
    printMatrix(c);
    std::cout << "res"<< endl;
    printMatrix(res);
    std::cout << "parallel"<< endl;
    printMatrix(paral);
}

int main() {
//
//    int deviceCount;
//    cudaDeviceProp deviceProp;
//
//    //Сколько устройств CUDA установлено на PC.
//    cudaGetDeviceCount(&deviceCount);
//
//    printf("Device count: %d\n\n", deviceCount);

    const size_t MATRIX_SIZE = 5000;
    const unsigned short int RUNS = 5;

    Matrix a = getRandArray(MATRIX_SIZE);
    Matrix b = getRandArray(MATRIX_SIZE);
    Matrix c = getRandArray(MATRIX_SIZE);
    Matrix res = getRandArray(MATRIX_SIZE);

    m_time globalTime;

    std::cout<<"start calculating..."<<std::endl;
    for (int i = 0; i< RUNS; ++i) {

        memoryType t = global;
        MatrixResult r = cudaCalc(a,b,c, res, t);

        globalTime.max = globalTime.max < r.time.total ? r.time.total : globalTime.max;
        globalTime.min = globalTime.min > r.time.total ? r.time.total: globalTime.min;
        globalTime.average += r.time.total;

    }
    globalTime.average = globalTime.average / RUNS;

    std::cout<< "Average time using global memory (ms): "<< globalTime.average<<std::endl;
    std::cout<< "Min time using global memory: "<< globalTime.min<<std::endl;
    std::cout<< "Max time using global memory: "<< globalTime.max<<std::endl<<std::endl;

    //shared
    m_time sharedTime;
    MatrixResult r;
    for (int i = 0; i< RUNS; ++i) {

        memoryType t = shared;
        r = cudaCalc(a,b,c, res, t);

        sharedTime.max = sharedTime.max < r.time.total ? r.time.total : sharedTime.max;
        sharedTime.min = sharedTime.min > r.time.total ? r.time.total: sharedTime.min;
        sharedTime.average += r.time.total;

    }
    sharedTime.average = sharedTime.average / RUNS;
    std::cout<<"copy to device "<< r.time.toDevice<<std::endl;
    std::cout<<"kernel "<< r.time.kernel<<std::endl;
    std::cout<<"copy to host "<< r.time.toHost<<std::endl;
    std::cout<< "Average time using shared memory (ms): "<< sharedTime.average<<std::endl;
    std::cout<< "Min time using shared memory: "<< sharedTime.min<<std::endl;
    std::cout<< "Max time using shared memory: "<< sharedTime.max<<std::endl;

    if (MATRIX_SIZE < 100) {
        std::cout<<"matrix A"<<std::endl;
        printMatrix(a);
        std::cout<<"matrix B"<<std::endl;
        printMatrix(b);
        std::cout<<"matrix C"<<std::endl;
        printMatrix(c);
        std::cout<<"result"<<std::endl;
        printMatrix(res);
    }


    return 0;
}

