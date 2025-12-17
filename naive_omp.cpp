// naive_omp.cpp
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/time.h>

void mat_mult_omp(double *A, double *B, double *C, int N) {
    // Parallelize the outer loop
    #pragma omp parallel for collapse(1) schedule(static)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main(int argc, char *argv[]) {
    int N = 10; // Default size, change as needed for scalability study 
    if (argc > 1) N = atoi(argv[1]);

    // Allocation
    double *A = (double*)malloc(N * N * sizeof(double));
    double *B = (double*)malloc(N * N * sizeof(double));
    double *C = (double*)malloc(N * N * sizeof(double));

    // Initialize (example data)
    for(int i=0; i<N*N; i++) {
        A[i] = 1.0;
        B[i] = 1.0; 
    }

    double start = omp_get_wtime();
    mat_mult_omp(A, B, C, N);
    double end = omp_get_wtime();

    printf("OpenMP Naive Matrix Mult (N=%d) Time: %f seconds\n", N, end - start);

    free(A); free(B); free(C);
    return 0;
}