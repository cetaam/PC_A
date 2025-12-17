#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h> // Required for fabs() in verification

// 1. Sequential multiply function (Kernel)
void sequential_matrix_mult(double *local_A, double *B, double *local_C, int local_rows, int N) {
    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += local_A[i * N + k] * B[k * N + j];
            }
            local_C[i * N + j] = sum;
        }
    }
}

// 2. Verification function
int verify_result(double *C, int N) {
    int correct = 1;
    for (int i = 0; i < N * N; i++) {
        // Since A and B are initialized to 1.0, every element in C must be N
        if (fabs(C[i] - N) > 1e-9) { 
            correct = 0;
            break;
        }
    }
    return correct;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 10; // Default size
    if (argc > 1) {
        N = atoi(argv[1]);
    }

    // --- PART 1: Calculate Uneven Loads ---
    int base_rows = N / size;
    int remainder = N % size;

    // Arrays to store how many items each process sends/receives
    int *sendcounts = (int *)malloc(size * sizeof(int));
    int *displs = (int *)malloc(size * sizeof(int));

    int offset = 0;
    for (int i = 0; i < size; i++) {
        // Distribute remainder rows to the first few processes
        int rows = base_rows + (i < remainder ? 1 : 0);
        
        // We are sending/receiving doubles, so count = rows * N
        sendcounts[i] = rows * N;
        displs[i] = offset;
        offset += sendcounts[i];
    }

    // Determine how many rows THIS process is responsible for
    int my_rows = sendcounts[rank] / N;

    printf("Process %d received %d rows.\n", rank, my_rows);


    // --- PART 2: Memory Allocation ---
    double *A = NULL, *B = NULL, *C = NULL;
    double *local_A = (double *)malloc(my_rows * N * sizeof(double));
    double *local_C = (double *)malloc(my_rows * N * sizeof(double));

    if (rank == 0) {
        A = (double *)malloc(N * N * sizeof(double));
        B = (double *)malloc(N * N * sizeof(double));
        C = (double *)malloc(N * N * sizeof(double));

        // Initialize A and B
        for (int i = 0; i < N * N; i++) {
            A[i] = 1.0; 
            B[i] = 1.0; 
        }
    }

    // --- PART 3: Evaluation Start ---
    // Ensure everyone is ready before starting the clock
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    // Broadcast B to everyone (Everyone needs the full B matrix)
    B = rank == 0 ? B : (double *)malloc(N * N * sizeof(double));
    MPI_Bcast(B, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Scatter A (Variable sizes)
    MPI_Scatterv(A, sendcounts, displs, MPI_DOUBLE, 
                 local_A, my_rows * N, MPI_DOUBLE, 
                 0, MPI_COMM_WORLD);

    // Compute
    sequential_matrix_mult(local_A, B, local_C, my_rows, N);

    // Gather C (Variable sizes)
    MPI_Gatherv(local_C, my_rows * N, MPI_DOUBLE, 
                C, sendcounts, displs, MPI_DOUBLE, 
                0, MPI_COMM_WORLD);

    // Stop Clock
    double end_time = MPI_Wtime();
    // --- PART 4: Evaluation End ---


    // --- PART 5: Reporting ---
    if (rank == 0) {
        double elapsed_time = end_time - start_time;
        
        printf("------------------------------------------------\n");
        printf("Matrix Size (N)   : %d\n", N);
        printf("Processes (np)    : %d\n", size);
        printf("Time Taken        : %f seconds\n", elapsed_time);

        if (verify_result(C, N)) {
            printf("Result Verification: PASSED\n");
        } else {
            printf("Result Verification: FAILED\n");
        }
        printf("------------------------------------------------\n");

        free(A); free(B); free(C);
    }

    // Cleanup
    free(local_A);
    free(local_C);
    free(sendcounts);
    free(displs);
    if (rank != 0) {
        free(B);
    }

    MPI_Finalize();
    return 0;
}