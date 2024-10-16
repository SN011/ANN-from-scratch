#include "SYCLMatrixLib.h"
#include "1DMatrixLib.h"
#include <omp.h>
#include <iostream>

int main() {
    int N = 512;
    Matrix A(N, N);
    Matrix B(N, N);
    A.RandInit();
    B.RandInit();
    

    double start = omp_get_wtime();
    A.multiply(B);
    std::cout << "took " << omp_get_wtime() - start << std::endl;
    
    Matrix1D M(N, N);
    Matrix1D n(N, N);
    M.RandInit();
    n.RandInit();
    
    start = omp_get_wtime();
    M.multiply(n);
    std::cout << "took " << omp_get_wtime() - start << std::endl;

    return 0;
}
