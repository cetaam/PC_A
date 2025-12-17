// naive_ompi.cpp (Hybrid MPI + OpenMP)
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include "matrix_io.h"

static void hybrid_matrix_mult(const double *local_A, const double *B,
                               double *local_C, int local_rows, int N) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < N; j++) local_C[i*N + j] = 0.0;

        for (int k = 0; k < N; k++) {
            double r = local_A[i*N + k];
            const double *b = B + k*N;
            #pragma omp simd
            for (int j = 0; j < N; j++) local_C[i*N + j] += r * b[j];
        }
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank=0, size=1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int expected_N = -1;
    if (argc > 1) expected_N = atoi(argv[1]);
    const char *input_file = (argc > 2) ? argv[2] : "matrix_input.txt";

    int N=0;
    double *A=NULL, *B=NULL, *C=NULL;

    if (rank == 0) {
        if (read_matrix_file(input_file, &N, &A, &B) != 0) MPI_Abort(MPI_COMM_WORLD, 1);
        if (expected_N > 0 && expected_N != N) MPI_Abort(MPI_COMM_WORLD, 1);
        C = (double*)malloc((size_t)N*(size_t)N*sizeof(double));
        if (!C) MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        B = (double*)malloc((size_t)N*(size_t)N*sizeof(double));
        if (!B) MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rows_per = N / size;
    int rem = N % size;
    int my_rows = rows_per + (rank < rem ? 1 : 0);

    int *sendcounts = (int*)malloc((size_t)size*sizeof(int));
    int *displs     = (int*)malloc((size_t)size*sizeof(int));
    if (!sendcounts || !displs) MPI_Abort(MPI_COMM_WORLD, 1);

    int offset = 0;
    for (int r = 0; r < size; r++) {
        int rows = rows_per + (r < rem ? 1 : 0);
        sendcounts[r] = rows * N;
        displs[r] = offset;
        offset += sendcounts[r];
    }

    double *local_A = (double*)malloc((size_t)my_rows*(size_t)N*sizeof(double));
    double *local_C = (double*)malloc((size_t)my_rows*(size_t)N*sizeof(double));
    if (!local_A || !local_C) MPI_Abort(MPI_COMM_WORLD, 1);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    MPI_Bcast(B, N*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Scatterv(A, sendcounts, displs, MPI_DOUBLE,
                 local_A, my_rows*N, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    hybrid_matrix_mult(local_A, B, local_C, my_rows, N);

    MPI_Gatherv(local_C, my_rows*N, MPI_DOUBLE,
                C, sendcounts, displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    if (rank == 0) {
        printf("Hybrid Naive (MPI+OpenMP) (N=%d, np=%d, threads=%d) Time: %f s\n",
               N, size, omp_get_max_threads(), t1 - t0);
        verify_or_warn("[naive_ompi]", A, B, C, N, N);
        free(A); free(B); free(C);
    } else {
        free(B);
    }

    free(local_A); free(local_C);
    free(sendcounts); free(displs);

    MPI_Finalize();
    return 0;
}
