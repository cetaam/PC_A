#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "matrix_io.h"

// -------- Tuning --------
#define CUTOFF 256          // base case for serial Strassen (naive)
#define MAX_PAD 8192        // safety: avoid padding explosion (10000 -> 16384)

static inline int is_power_of_two(int x) {
    return (x > 0) && ((x & (x - 1)) == 0);
}
static inline int next_power_of_two(int x) {
    if (x <= 1) return 1;
    int p = 1;
    while (p < x) p <<= 1;
    return p;
}

static void die_if_null(void *p, const char *msg) {
    if (!p) {
        fprintf(stderr, "Error: %s\n", msg);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

// -------- Workspace (arena) for serial Strassen --------
typedef struct {
    double *base;
    size_t cap;  // doubles
    size_t top;  // doubles
} Arena;

static inline size_t arena_mark(Arena *a) { return a->top; }
static inline void arena_release(Arena *a, size_t mark) { a->top = mark; }

static inline double* arena_alloc(Arena *a, size_t n_doubles) {
    if (a->top + n_doubles > a->cap) {
        fprintf(stderr, "Error: arena exhausted (need %zu doubles, cap=%zu)\n",
                a->top + n_doubles, a->cap);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    double *p = a->base + a->top;
    a->top += n_doubles;
    return p;
}

// conservative, safe bound for a serial subtree
static inline size_t arena_needed_doubles(int n) {
    return (size_t)20 * (size_t)n * (size_t)n;
}

// -------- View ops (row-major with leading dimension) --------
static inline void add_view(const double *A, int lda,
                            const double *B, int ldb,
                            double *R, int ldr,
                            int n) {
    for (int i = 0; i < n; i++) {
        const double *a = A + i * lda;
        const double *b = B + i * ldb;
        double *r = R + i * ldr;
        for (int j = 0; j < n; j++) r[j] = a[j] + b[j];
    }
}

static inline void sub_view(const double *A, int lda,
                            const double *B, int ldb,
                            double *R, int ldr,
                            int n) {
    for (int i = 0; i < n; i++) {
        const double *a = A + i * lda;
        const double *b = B + i * ldb;
        double *r = R + i * ldr;
        for (int j = 0; j < n; j++) r[j] = a[j] - b[j];
    }
}

static inline void zero_view(double *C, int ldc, int n) {
    for (int i = 0; i < n; i++) {
        memset(C + i * ldc, 0, (size_t)n * sizeof(double));
    }
}

static inline void axpy_view(double alpha,
                             const double *X, int ldx,
                             double *C, int ldc,
                             int n) {
    for (int i = 0; i < n; i++) {
        const double *x = X + i * ldx;
        double *c = C + i * ldc;
        for (int j = 0; j < n; j++) c[j] += alpha * x[j];
    }
}

// -------- Naive multiply: C = A*B (views) --------
static void naive_mult_view(const double *A, int lda,
                            const double *B, int ldb,
                            double *C, int ldc,
                            int n) {
    for (int i = 0; i < n; i++) {
        double *c = C + i * ldc;
        for (int j = 0; j < n; j++) c[j] = 0.0;

        const double *a = A + i * lda;
        for (int k = 0; k < n; k++) {
            double r = a[k];
            const double *b = B + k * ldb;
            for (int j = 0; j < n; j++) {
                c[j] += r * b[j];
            }
        }
    }
}

// -------- Serial Strassen (no MPI, no OMP) --------
static void strassen_serial(const double *A, int lda,
                            const double *B, int ldb,
                            double *C, int ldc,
                            int n,
                            Arena *arena) {
    if (n <= CUTOFF) {
        naive_mult_view(A, lda, B, ldb, C, ldc, n);
        return;
    }

    int k = n / 2;
    size_t kk = (size_t)k * (size_t)k;

    const double *A11 = A;
    const double *A12 = A + k;
    const double *A21 = A + k * lda;
    const double *A22 = A + k * lda + k;

    const double *B11 = B;
    const double *B12 = B + k;
    const double *B21 = B + k * ldb;
    const double *B22 = B + k * ldb + k;

    double *C11 = C;
    double *C12 = C + k;
    double *C21 = C + k * ldc;
    double *C22 = C + k * ldc + k;

    size_t mark = arena_mark(arena);

    double *M1 = arena_alloc(arena, kk);
    double *M2 = arena_alloc(arena, kk);
    double *M3 = arena_alloc(arena, kk);
    double *M4 = arena_alloc(arena, kk);
    double *M5 = arena_alloc(arena, kk);
    double *M6 = arena_alloc(arena, kk);
    double *M7 = arena_alloc(arena, kk);
    double *T1 = arena_alloc(arena, kk);
    double *T2 = arena_alloc(arena, kk);

    // M1 = (A11+A22)(B11+B22)
    add_view(A11, lda, A22, lda, T1, k, k);
    add_view(B11, ldb, B22, ldb, T2, k, k);
    strassen_serial(T1, k, T2, k, M1, k, k, arena);

    // M2 = (A21+A22)B11
    add_view(A21, lda, A22, lda, T1, k, k);
    strassen_serial(T1, k, B11, ldb, M2, k, k, arena);

    // M3 = A11(B12-B22)
    sub_view(B12, ldb, B22, ldb, T2, k, k);
    strassen_serial(A11, lda, T2, k, M3, k, k, arena);

    // M4 = A22(B21-B11)
    sub_view(B21, ldb, B11, ldb, T2, k, k);
    strassen_serial(A22, lda, T2, k, M4, k, k, arena);

    // M5 = (A11+A12)B22
    add_view(A11, lda, A12, lda, T1, k, k);
    strassen_serial(T1, k, B22, ldb, M5, k, k, arena);

    // M6 = (A21-A11)(B11+B12)
    sub_view(A21, lda, A11, lda, T1, k, k);
    add_view(B11, ldb, B12, ldb, T2, k, k);
    strassen_serial(T1, k, T2, k, M6, k, k, arena);

    // M7 = (A12-A22)(B21+B22)
    sub_view(A12, lda, A22, lda, T1, k, k);
    add_view(B21, ldb, B22, ldb, T2, k, k);
    strassen_serial(T1, k, T2, k, M7, k, k, arena);

    // Combine
    zero_view(C11, ldc, k);
    zero_view(C12, ldc, k);
    zero_view(C21, ldc, k);
    zero_view(C22, ldc, k);

    axpy_view(+1.0, M1, k, C11, ldc, k);
    axpy_view(+1.0, M4, k, C11, ldc, k);
    axpy_view(-1.0, M5, k, C11, ldc, k);
    axpy_view(+1.0, M7, k, C11, ldc, k);

    axpy_view(+1.0, M3, k, C12, ldc, k);
    axpy_view(+1.0, M5, k, C12, ldc, k);

    axpy_view(+1.0, M2, k, C21, ldc, k);
    axpy_view(+1.0, M4, k, C21, ldc, k);

    axpy_view(+1.0, M1, k, C22, ldc, k);
    axpy_view(-1.0, M2, k, C22, ldc, k);
    axpy_view(+1.0, M3, k, C22, ldc, k);
    axpy_view(+1.0, M6, k, C22, ldc, k);

    arena_release(arena, mark);
}

// Compute one Strassen Mi locally (Mi_id in 1..7) using broadcast A,B
static void compute_Mi(int Mi_id,
                       const double *A, const double *B, int N,  // A,B are N x N with ld=N
                       double *Mi_out, int k,                    // Mi_out is k x k contiguous (ld=k)
                       Arena *arena) {

    const double *A11 = A;
    const double *A12 = A + k;
    const double *A21 = A + k * N;
    const double *A22 = A + k * N + k;

    const double *B11 = B;
    const double *B12 = B + k;
    const double *B21 = B + k * N;
    const double *B22 = B + k * N + k;

    size_t kk = (size_t)k * (size_t)k;
    size_t mark = arena_mark(arena);

    double *T1 = arena_alloc(arena, kk);
    double *T2 = arena_alloc(arena, kk);

    switch (Mi_id) {
        case 1: // (A11+A22)(B11+B22)
            add_view(A11, N, A22, N, T1, k, k);
            add_view(B11, N, B22, N, T2, k, k);
            strassen_serial(T1, k, T2, k, Mi_out, k, k, arena);
            break;
        case 2: // (A21+A22)B11
            add_view(A21, N, A22, N, T1, k, k);
            strassen_serial(T1, k, B11, N, Mi_out, k, k, arena);
            break;
        case 3: // A11(B12-B22)
            sub_view(B12, N, B22, N, T2, k, k);
            strassen_serial(A11, N, T2, k, Mi_out, k, k, arena);
            break;
        case 4: // A22(B21-B11)
            sub_view(B21, N, B11, N, T2, k, k);
            strassen_serial(A22, N, T2, k, Mi_out, k, k, arena);
            break;
        case 5: // (A11+A12)B22
            add_view(A11, N, A12, N, T1, k, k);
            strassen_serial(T1, k, B22, N, Mi_out, k, k, arena);
            break;
        case 6: // (A21-A11)(B11+B12)
            sub_view(A21, N, A11, N, T1, k, k);
            add_view(B11, N, B12, N, T2, k, k);
            strassen_serial(T1, k, T2, k, Mi_out, k, k, arena);
            break;
        case 7: // (A12-A22)(B21+B22)
            sub_view(A12, N, A22, N, T1, k, k);
            add_view(B21, N, B22, N, T2, k, k);
            strassen_serial(T1, k, T2, k, Mi_out, k, k, arena);
            break;
        default:
            fprintf(stderr, "Invalid Mi_id=%d\n", Mi_id);
            MPI_Abort(MPI_COMM_WORLD, 1);
    }

    arena_release(arena, mark);
}

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // CLI: ./strassen_mpi [expected_N] [input_file]
    int expected_N = -1;
    if (argc > 1) expected_N = atoi(argv[1]);
    const char *input_file = (argc > 2) ? argv[2] : "matrix_input.txt";

    int N = 0;
    double *A_in = NULL, *B_in = NULL;

    if (rank == 0) {
        if (read_matrix_file(input_file, &N, &A_in, &B_in) != 0) {
            fprintf(stderr, "Rank 0: failed to read '%s'\n", input_file);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (expected_N > 0 && expected_N != N) {
            fprintf(stderr, "Rank 0: expected N=%d but file has N=%d\n", expected_N, N);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int Npad = is_power_of_two(N) ? N : next_power_of_two(N);
    if (Npad > MAX_PAD) {
        if (rank == 0) {
            fprintf(stderr,
                    "Strassen MPI: N=%d pads to %d (too large). Skipping to avoid OOM.\n",
                    N, Npad);
        }
        if (rank == 0) { free(A_in); free(B_in); }
        MPI_Finalize();
        return 2;
    }

    if (rank == 0 && Npad != N) {
        printf("Note: padding N=%d up to Npad=%d for Strassen MPI correctness.\n", N, Npad);
    }

    // Allocate padded A,B on all ranks
    size_t NN = (size_t)Npad * (size_t)Npad;
    double *A = (double*)calloc(NN, sizeof(double));
    double *B = (double*)calloc(NN, sizeof(double));
    double *C = NULL;
    die_if_null(A, "alloc A failed");
    die_if_null(B, "alloc B failed");

    if (rank == 0) {
        // copy N x N into top-left of padded
        for (int i = 0; i < N; i++) {
            memcpy(A + (size_t)i * Npad, A_in + (size_t)i * N, (size_t)N * sizeof(double));
            memcpy(B + (size_t)i * Npad, B_in + (size_t)i * N, (size_t)N * sizeof(double));
        }
        free(A_in); free(B_in);

        C = (double*)malloc(NN * sizeof(double));
        die_if_null(C, "alloc C failed");
        memset(C, 0, NN * sizeof(double));
    }

    // Broadcast padded A and B to everyone (simple + robust)
    MPI_Bcast(A, (int)NN, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(B, (int)NN, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    // Top-level Strassen: compute M1..M7 in parallel across ranks
    int k = Npad / 2;
    size_t kk = (size_t)k * (size_t)k;

    // Each rank keeps buffers for Mi it computes; others keep zeros.
    double *Mlocal = (double*)calloc(kk, sizeof(double));
    die_if_null(Mlocal, "alloc Mlocal failed");

    // Arena for local serial Strassen subtree on size k
    Arena arena;
    arena.cap = arena_needed_doubles(k) + (size_t)4 * kk; // extra slack
    arena.base = (double*)malloc(arena.cap * sizeof(double));
    die_if_null(arena.base, "alloc arena failed");
    arena.top = 0;

    // Root will reduce each Mi separately (sum); exactly one rank contributes.
    double *Mroot = NULL;
    if (rank == 0) {
        Mroot = (double*)malloc(kk * sizeof(double));
        die_if_null(Mroot, "alloc Mroot failed");
    }

    // We distribute Mi_id 1..7 round-robin across ranks:
    // rank r computes Mi where (Mi_id-1) % size == r
    // Then we reduce it to root. Repeat for each Mi_id.
    double *M1 = NULL, *M2 = NULL, *M3 = NULL, *M4 = NULL, *M5 = NULL, *M6 = NULL, *M7 = NULL;
    if (rank == 0) {
        M1 = (double*)malloc(kk * sizeof(double));
        M2 = (double*)malloc(kk * sizeof(double));
        M3 = (double*)malloc(kk * sizeof(double));
        M4 = (double*)malloc(kk * sizeof(double));
        M5 = (double*)malloc(kk * sizeof(double));
        M6 = (double*)malloc(kk * sizeof(double));
        M7 = (double*)malloc(kk * sizeof(double));
        die_if_null(M1,"alloc M1"); die_if_null(M2,"alloc M2"); die_if_null(M3,"alloc M3");
        die_if_null(M4,"alloc M4"); die_if_null(M5,"alloc M5"); die_if_null(M6,"alloc M6"); die_if_null(M7,"alloc M7");
    }

    for (int Mi_id = 1; Mi_id <= 7; Mi_id++) {
        memset(Mlocal, 0, kk * sizeof(double));
        arena.top = 0;

        if (((Mi_id - 1) % size) == rank) {
            compute_Mi(Mi_id, A, B, Npad, Mlocal, k, &arena);
        }

        MPI_Reduce(Mlocal, Mroot, (int)kk, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            // copy into the right M buffer
            switch (Mi_id) {
                case 1: memcpy(M1, Mroot, kk * sizeof(double)); break;
                case 2: memcpy(M2, Mroot, kk * sizeof(double)); break;
                case 3: memcpy(M3, Mroot, kk * sizeof(double)); break;
                case 4: memcpy(M4, Mroot, kk * sizeof(double)); break;
                case 5: memcpy(M5, Mroot, kk * sizeof(double)); break;
                case 6: memcpy(M6, Mroot, kk * sizeof(double)); break;
                case 7: memcpy(M7, Mroot, kk * sizeof(double)); break;
            }
        }
    }

    // Root combines into C
    if (rank == 0) {
        double *C11 = C;
        double *C12 = C + k;
        double *C21 = C + k * Npad;
        double *C22 = C + k * Npad + k;

        zero_view(C11, Npad, k);
        zero_view(C12, Npad, k);
        zero_view(C21, Npad, k);
        zero_view(C22, Npad, k);

        // C11 = M1 + M4 - M5 + M7
        axpy_view(+1.0, M1, k, C11, Npad, k);
        axpy_view(+1.0, M4, k, C11, Npad, k);
        axpy_view(-1.0, M5, k, C11, Npad, k);
        axpy_view(+1.0, M7, k, C11, Npad, k);

        // C12 = M3 + M5
        axpy_view(+1.0, M3, k, C12, Npad, k);
        axpy_view(+1.0, M5, k, C12, Npad, k);

        // C21 = M2 + M4
        axpy_view(+1.0, M2, k, C21, Npad, k);
        axpy_view(+1.0, M4, k, C21, Npad, k);

        // C22 = M1 - M2 + M3 + M6
        axpy_view(+1.0, M1, k, C22, Npad, k);
        axpy_view(-1.0, M2, k, C22, Npad, k);
        axpy_view(+1.0, M3, k, C22, Npad, k);
        axpy_view(+1.0, M6, k, C22, Npad, k);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    if (rank == 0) {
        printf("MPI Strassen (top-level distributed) (N=%d, padded=%d, np=%d) Time: %f s\n",
               N, Npad, size, t1 - t0);
               verify_or_warn("[strassen_mpi]", A, B, C, N, Npad);
    }

    free(arena.base);
    free(Mlocal);
    if (rank == 0) {
        free(Mroot);
        free(M1); free(M2); free(M3); free(M4); free(M5); free(M6); free(M7);
        free(C);
        
    }
    free(A);
    free(B);

    MPI_Finalize();
    return 0;
}
