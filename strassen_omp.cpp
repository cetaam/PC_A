#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Tuning: Switch to standard O(N^3) multiplication when matrix is small
// This prevents overhead from creating too many tiny tasks
#define CUTOFF 64 

// Standard Matrix Multiply (Row-Major)
void naive_mult(double *A, double *B, double *C, int N) {
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            double temp = A[i * N + k];
            for (int j = 0; j < N; j++) {
                C[i * N + j] += temp * B[k * N + j];
            }
        }
    }
}

void add(double *A, double *B, double *C, int size) {
    for (int i = 0; i < size * size; i++) C[i] = A[i] + B[i];
}

void sub(double *A, double *B, double *C, int size) {
    for (int i = 0; i < size * size; i++) C[i] = A[i] - B[i];
}

// Helper to manage memory allocation for sub-matrices
double* allocate_matrix(int size) {
    return (double*)calloc(size * size, sizeof(double));
}

void strassen(double *A, double *B, double *C, int N) {
    // 1. Base Case: Use naive multiplication for small matrices
    if (N <= CUTOFF) {
        naive_mult(A, B, C, N);
        return;
    }

    int k = N / 2;
    int sz = k * k; // Size of sub-matrix in elements

    // 2. Allocate Sub-matrices
    // We allocate NEW memory here to avoid Race Conditions.
    // Each thread/task gets its own unique memory space.
    double *A11 = allocate_matrix(k); double *A12 = allocate_matrix(k);
    double *A21 = allocate_matrix(k); double *A22 = allocate_matrix(k);
    double *B11 = allocate_matrix(k); double *B12 = allocate_matrix(k);
    double *B21 = allocate_matrix(k); double *B22 = allocate_matrix(k);

    double *M1 = allocate_matrix(k); double *M2 = allocate_matrix(k);
    double *M3 = allocate_matrix(k); double *M4 = allocate_matrix(k);
    double *M5 = allocate_matrix(k); double *M6 = allocate_matrix(k);
    double *M7 = allocate_matrix(k);

    double *T1 = allocate_matrix(k); double *T2 = allocate_matrix(k);

    // 3. Split input matrices A and B into quadrants
    // We copy data to ensure contiguous memory layout for recursion
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            A11[i * k + j] = A[i * N + j];
            A12[i * k + j] = A[i * N + (j + k)];
            A21[i * k + j] = A[(i + k) * N + j];
            A22[i * k + j] = A[(i + k) * N + (j + k)];

            B11[i * k + j] = B[i * N + j];
            B12[i * k + j] = B[i * N + (j + k)];
            B21[i * k + j] = B[(i + k) * N + j];
            B22[i * k + j] = B[(i + k) * N + (j + k)];
        }
    }

    // 4. Compute M1-M7 Parallel Tasks
    // #pragma omp task creates a new task that can be picked up by any thread
    
    #pragma omp task shared(M1) firstprivate(T1, T2) // T1/T2 need to be private or re-allocated!
    {
        // To be truly safe, better to allocate temps inside task or strictly separate
        // For simplicity here, we do calculation serially inside the task or alloc inside
        double *t1 = allocate_matrix(k); 
        double *t2 = allocate_matrix(k);
        add(A11, A22, t1, k);
        add(B11, B22, t2, k);
        strassen(t1, t2, M1, k); 
        free(t1); free(t2);
    }

    #pragma omp task shared(M2)
    {
        double *t1 = allocate_matrix(k);
        add(A21, A22, t1, k);
        strassen(t1, B11, M2, k);
        free(t1);
    }

    #pragma omp task shared(M3)
    {
        double *t1 = allocate_matrix(k);
        sub(B12, B22, t1, k);
        strassen(A11, t1, M3, k);
        free(t1);
    }

    #pragma omp task shared(M4)
    {
        double *t1 = allocate_matrix(k);
        sub(B21, B11, t1, k);
        strassen(A22, t1, M4, k);
        free(t1);
    }

    #pragma omp task shared(M5)
    {
        double *t1 = allocate_matrix(k);
        add(A11, A12, t1, k);
        strassen(t1, B22, M5, k);
        free(t1);
    }

    #pragma omp task shared(M6)
    {
        double *t1 = allocate_matrix(k);
        double *t2 = allocate_matrix(k);
        sub(A21, A11, t1, k);
        add(B11, B12, t2, k);
        strassen(t1, t2, M6, k);
        free(t1); free(t2);
    }

    #pragma omp task shared(M7)
    {
        double *t1 = allocate_matrix(k);
        double *t2 = allocate_matrix(k);
        sub(A12, A22, t1, k);
        add(B21, B22, t2, k);
        strassen(t1, t2, M7, k);
        free(t1); free(t2);
    }

    // Wait for all 7 tasks to finish before combining
    #pragma omp taskwait

    // 5. Combine results into C
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            C[i * N + j]           = M1[i*k+j] + M4[i*k+j] - M5[i*k+j] + M7[i*k+j]; // C11
            C[i * N + (j + k)]     = M3[i*k+j] + M5[i*k+j];                         // C12
            C[(i + k) * N + j]     = M2[i*k+j] + M4[i*k+j];                         // C21
            C[(i + k) * N + (j + k)] = M1[i*k+j] - M2[i*k+j] + M3[i*k+j] + M6[i*k+j]; // C22
        }
    }

    // 6. Cleanup Memory
    free(A11); free(A12); free(A21); free(A22);
    free(B11); free(B12); free(B21); free(B22);
    free(M1); free(M2); free(M3); free(M4); free(M5); free(M6); free(M7);
    free(T1); free(T2);
}

int main(int argc, char *argv[]) {
    int N = 1024; // Power of 2
    if(argc > 1) N = atoi(argv[1]);

    // Allocate Main Matrices
    double *A = (double*)malloc(N * N * sizeof(double));
    double *B = (double*)malloc(N * N * sizeof(double));
    double *C = (double*)calloc(N * N, sizeof(double));

    // Initialize
    for(int i=0; i<N*N; i++) { A[i] = 1.0; B[i] = 1.0; }

    double start = omp_get_wtime();

    // Start Parallel Region
    #pragma omp parallel
    {
        // Only one thread starts the recursion, others join via tasks
        #pragma omp single
        {
            strassen(A, B, C, N);
        }
    }

    double end = omp_get_wtime();
    printf("Strassen OpenMP (N=%d) Time: %f s\n", N, end - start);
    printf("Verification C[0]: %f (Should be %d)\n", C[0], N);

    free(A); free(B); free(C);
    return 0;
}