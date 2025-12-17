#include <cblas.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

// Hàm đo thời gian chính xác trên Linux/WSL
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void random_matrix(int n, double *A) {
    for (int i = 0; i < n * n; i++) {
        A[i] = 1.0;
    }
}

int main() {
    int sizes[] = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024
                    , 2048, 4096, 8192, 10000};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    printf("Running benchmark using OpenBLAS on WSL...\n");
    printf("%-10s | %-15s | %-15s\n", "Size (N)", "Time (s)", "GFLOPS");
    printf("--------------------------------------------------\n");

    for (int k = 0; k < num_sizes; k++) {
        int n = sizes[k];
        
        double *A = (double*)malloc(n * n * sizeof(double));
        double *B = (double*)malloc(n * n * sizeof(double));
        double *C = (double*)malloc(n * n * sizeof(double));

        random_matrix(n, A);
        random_matrix(n, B);

        double start = get_time();
        
        // Gọi hàm nhân ma trận của OpenBLAS
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                    n, n, n, 1.0, A, n, B, n, 0.0, C, n);

        double end = get_time();
        
        double time_spent = end - start;
        double gflops = (2.0 * n * n * n) / time_spent / 1e9;

        printf("%-10d | %-15.6f | %-15.6f\n", n, time_spent, gflops);

        free(A); free(B); free(C);
    }
    return 0;
}