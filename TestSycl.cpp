#include "SYCLMatrixLib.h"
#include "1DMatrixLib.h"
#include <omp.h>
#include <iostream>

int main() {
    Matrix A(1024, 64);
    Matrix B(64, 32);
    A.RandInit();
    B.RandInit();
    

    double start = omp_get_wtime();
    A.multiply(B);
    std::cout << "took " << omp_get_wtime() - start << std::endl;
    
    Matrix1D M(1024, 64);
    Matrix1D N(64, 32);
    M.RandInit();
    N.RandInit();
    
    start = omp_get_wtime();
    M.multiply(N);
    std::cout << "took " << omp_get_wtime() - start << std::endl;

    return 0;
}
