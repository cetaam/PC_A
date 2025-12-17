// naive_omp.cpp
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#include "matrix_io.h"

void mat_mult_omp(const double *A, const double *B, double *C, int N)
{
    // Parallelize the outer loop (each thread works on different rows i)
#pragma omp parallel for schedule(static)
for (int i = 0; i < N; i++) {
  for (int j = 0; j < N; j++) C[i*N+j] = 0.0;

  for (int k = 0; k < N; k++) {
    double r = A[i*N+k];
     #pragma omp simd
    for (int j = 0; j < N; j++) {
      C[i*N+j] += r * B[k*N+j];
    }
  }
}

}

int main(int argc, char *argv[]) {

    int expected_N = -1;
    if (argc > 1) expected_N = atoi(argv[1]);

    const char *input_file = (argc > 2) ? argv[2] : "matrix_input.txt";

    double *A = NULL, *B = NULL;

    int N_file = 0;
    if (read_matrix_file(input_file, &N_file, &A, &B) != 0) {
        return -1;
    }

    if (expected_N > 0 && expected_N != N_file) {
        fprintf(stderr, "Error: expected N=%d but file '%s' contains N=%d\n",
                expected_N, input_file, N_file);
        free(A);
        free(B);
        return -1;
    }

    int N = N_file;

    double *C = (double *)malloc(N * N * sizeof(double));
    if (!C) {
        fprintf(stderr, "Error: malloc failed for C\n");
        free(A);
        free(B);
        return -1;
    }

    double start = omp_get_wtime();
    mat_mult_omp(A, B, C, N);
    double end = omp_get_wtime();

    printf("OpenMP Naive Matrix Mult (N=%d) Time: %f seconds\n", N, end - start);
      verify_or_warn("[naive_omp]", A, B, C, N, N);
    free(A);
    free(B);
    free(C);
    return 0;
}
