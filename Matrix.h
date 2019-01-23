
#ifndef CUDA_MATRIX_MATRIX_H
#define CUDA_MATRIX_MATRIX_H

#include <vector>
using namespace std;

enum memoryType {
    def = 1,
    global,
    shared,
    constant
};

struct Matrix {
    int* data;
    size_t columns;
    size_t strings;
};

struct gpuTime {
   float toDevice;
   float toHost;
   float kernel;
   float total;
};
struct MatrixResult {
    MatrixResult( Matrix& m , gpuTime& t) {
        matrix = m;
        time = t;
    }
    MatrixResult(){
        time = {0.0,0.0,0.0,0.0};
    }

    Matrix matrix;
    gpuTime time;
};

Matrix getRandArray(size_t str, size_t col);
Matrix getRandArray(size_t size);
Matrix createMatrix(size_t str, size_t col);
void printMatrix(Matrix m);
//MatrixResult constCalculate(Matrix a, Matrix b, Matrix c);
//void sharedCalculate(Matrix a, Matrix b, Matrix c);
//__global__ void globalCalculate(Matrix a, Matrix b, Matrix c);
MatrixResult cudaCalc(Matrix& a, Matrix& b, Matrix &c, Matrix& result, memoryType type);

#endif
