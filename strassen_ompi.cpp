// strassen_ompi.cpp  (MPI + OpenMP hybrid Strassen)
//
// MPI: distribute top-level M1..M7 across ranks
// OMP: each rank computes its assigned Mi using OpenMP tasks (top level only)

#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "matrix_io.h"

// -------- Tuning --------
#define CUTOFF     128    // base-case threshold (naive)
#define TASK_CUTOFF 512   // only create tasks above this size
#define PAR_LEVELS  1     // parallelize tasks only this many levels (1 = only top level)
#define MAX_PAD     8192  // safety: avoid padding explosion (e.g., 10000 -> 16384)

// ---------- Utilities ----------
static inline int is_power_of_two(int x) { return (x > 0) && ((x & (x - 1)) == 0); }
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

// ---------- Workspace (bump allocator) ----------
typedef struct {
    double *buf;
    size_t cap;  // doubles
    size_t top;  // doubles
} Workspace;

static inline size_t ws_mark(Workspace *ws) { return ws->top; }
static inline void ws_release(Workspace *ws, size_t mark) { ws->top = mark; }

static inline double* ws_alloc(Workspace *ws, size_t n_doubles) {
    if (ws->top + n_doubles > ws->cap) {
        fprintf(stderr, "Error: workspace exhausted (need %zu doubles, cap=%zu)\n",
                ws->top + n_doubles, ws->cap);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    double *p = ws->buf + ws->top;
    ws->top += n_doubles;
    return p;
}

// very safe upper bound for PAR_LEVELS=1 serial subtrees
static inline size_t workspace_needed_doubles(int n) {
    return (size_t)20 * (size_t)n * (size_t)n;
}

// ---------- Matrix ops on views (row-major) ----------
static inline void add_view(const double *A, int lda,
                            const double *B, int ldb,
                            double *C, int ldc,
                            int n) {
    for (int i = 0; i < n; i++) {
        const double *a = A + i * lda;
        const double *b = B + i * ldb;
        double *c = C + i * ldc;
        for (int j = 0; j < n; j++) c[j] = a[j] + b[j];
    }
}

static inline void sub_view(const double *A, int lda,
                            const double *B, int ldb,
                            double *C, int ldc,
                            int n) {
    for (int i = 0; i < n; i++) {
        const double *a = A + i * lda;
        const double *b = B + i * ldb;
        double *c = C + i * ldc;
        for (int j = 0; j < n; j++) c[j] = a[j] - b[j];
    }
}

// fast naive: i-k-j (good locality), parallel over i (safe)
static inline void naive_mult_view_omp(const double *A, int lda,
                                      const double *B, int ldb,
                                      double *C, int ldc,
                                      int n) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        double *c = C + i * ldc;
        for (int j = 0; j < n; j++) c[j] = 0.0;

        const double *a = A + i * lda;
        for (int k = 0; k < n; k++) {
            double r = a[k];
            const double *b = B + k * ldb;
            #pragma omp simd
            for (int j = 0; j < n; j++) {
                c[j] += r * b[j];
            }
        }
    }
}

// C = C + alpha*X
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

// ---------- Strassen serial recursion (workspace, no tasks) ----------
static void strassen_serial(const double *A, int lda,
                            const double *B, int ldb,
                            double *C, int ldc,
                            int n,
                            Workspace *ws) {
    if (n <= CUTOFF) {
        naive_mult_view_omp(A, lda, B, ldb, C, ldc, n);
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

    size_t mark = ws_mark(ws);

    double *M1 = ws_alloc(ws, kk);
    double *M2 = ws_alloc(ws, kk);
    double *M3 = ws_alloc(ws, kk);
    double *M4 = ws_alloc(ws, kk);
    double *M5 = ws_alloc(ws, kk);
    double *M6 = ws_alloc(ws, kk);
    double *M7 = ws_alloc(ws, kk);

    double *T1 = ws_alloc(ws, kk);
    double *T2 = ws_alloc(ws, kk);

    const int ld = k;

    add_view(A11, lda, A22, lda, T1, ld, k);
    add_view(B11, ldb, B22, ldb, T2, ld, k);
    strassen_serial(T1, ld, T2, ld, M1, ld, k, ws);

    add_view(A21, lda, A22, lda, T1, ld, k);
    strassen_serial(T1, ld, B11, ldb, M2, ld, k, ws);

    sub_view(B12, ldb, B22, ldb, T2, ld, k);
    strassen_serial(A11, lda, T2, ld, M3, ld, k, ws);

    sub_view(B21, ldb, B11, ldb, T2, ld, k);
    strassen_serial(A22, lda, T2, ld, M4, ld, k, ws);

    add_view(A11, lda, A12, lda, T1, ld, k);
    strassen_serial(T1, ld, B22, ldb, M5, ld, k, ws);

    sub_view(A21, lda, A11, lda, T1, ld, k);
    add_view(B11, ldb, B12, ldb, T2, ld, k);
    strassen_serial(T1, ld, T2, ld, M6, ld, k, ws);

    sub_view(A12, lda, A22, lda, T1, ld, k);
    add_view(B21, ldb, B22, ldb, T2, ld, k);
    strassen_serial(T1, ld, T2, ld, M7, ld, k, ws);

    // zero C quadrants
    for (int i = 0; i < k; i++) {
        memset(C11 + i*ldc, 0, (size_t)k * sizeof(double));
        memset(C12 + i*ldc, 0, (size_t)k * sizeof(double));
        memset(C21 + i*ldc, 0, (size_t)k * sizeof(double));
        memset(C22 + i*ldc, 0, (size_t)k * sizeof(double));
    }

    axpy_view(+1.0, M1, ld, C11, ldc, k);
    axpy_view(+1.0, M4, ld, C11, ldc, k);
    axpy_view(-1.0, M5, ld, C11, ldc, k);
    axpy_view(+1.0, M7, ld, C11, ldc, k);

    axpy_view(+1.0, M3, ld, C12, ldc, k);
    axpy_view(+1.0, M5, ld, C12, ldc, k);

    axpy_view(+1.0, M2, ld, C21, ldc, k);
    axpy_view(+1.0, M4, ld, C21, ldc, k);

    axpy_view(+1.0, M1, ld, C22, ldc, k);
    axpy_view(-1.0, M2, ld, C22, ldc, k);
    axpy_view(+1.0, M3, ld, C22, ldc, k);
    axpy_view(+1.0, M6, ld, C22, ldc, k);

    ws_release(ws, mark);
}

// ---------- Parallel-at-top Strassen (tasks) ----------
static void strassen_parallel(const double *A, int lda,
                              const double *B, int ldb,
                              double *C, int ldc,
                              int n,
                              int depth) {
    if (n <= CUTOFF) {
        naive_mult_view_omp(A, lda, B, ldb, C, ldc, n);
        return;
    }

    if (depth >= PAR_LEVELS || n <= TASK_CUTOFF) {
        Workspace ws;
        ws.cap = workspace_needed_doubles(n);
        ws.buf = (double*)malloc(ws.cap * sizeof(double));
        die_if_null(ws.buf, "workspace malloc failed");
        ws.top = 0;

        strassen_serial(A, lda, B, ldb, C, ldc, n, &ws);

        free(ws.buf);
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

    double *block = (double*)calloc(7 * kk, sizeof(double));
    die_if_null(block, "M block alloc failed");

    double *M1 = block + 0 * kk;
    double *M2 = block + 1 * kk;
    double *M3 = block + 2 * kk;
    double *M4 = block + 3 * kk;
    double *M5 = block + 4 * kk;
    double *M6 = block + 5 * kk;
    double *M7 = block + 6 * kk;

    #pragma omp taskgroup
    {
        #pragma omp task shared(M1)
        {
            Workspace ws;
            ws.cap = workspace_needed_doubles(k);
            ws.buf = (double*)malloc(ws.cap * sizeof(double));
            die_if_null(ws.buf, "ws M1 malloc failed");
            ws.top = 0;

            double *T1 = ws_alloc(&ws, kk);
            double *T2 = ws_alloc(&ws, kk);
            add_view(A11, lda, A22, lda, T1, k, k);
            add_view(B11, ldb, B22, ldb, T2, k, k);
            strassen_parallel(T1, k, T2, k, M1, k, k, depth + 1);
            free(ws.buf);
        }

        #pragma omp task shared(M2)
        {
            Workspace ws;
            ws.cap = workspace_needed_doubles(k);
            ws.buf = (double*)malloc(ws.cap * sizeof(double));
            die_if_null(ws.buf, "ws M2 malloc failed");
            ws.top = 0;

            double *T1 = ws_alloc(&ws, kk);
            add_view(A21, lda, A22, lda, T1, k, k);
            strassen_parallel(T1, k, B11, ldb, M2, k, k, depth + 1);
            free(ws.buf);
        }

        #pragma omp task shared(M3)
        {
            Workspace ws;
            ws.cap = workspace_needed_doubles(k);
            ws.buf = (double*)malloc(ws.cap * sizeof(double));
            die_if_null(ws.buf, "ws M3 malloc failed");
            ws.top = 0;

            double *T2 = ws_alloc(&ws, kk);
            sub_view(B12, ldb, B22, ldb, T2, k, k);
            strassen_parallel(A11, lda, T2, k, M3, k, k, depth + 1);
            free(ws.buf);
        }

        #pragma omp task shared(M4)
        {
            Workspace ws;
            ws.cap = workspace_needed_doubles(k);
            ws.buf = (double*)malloc(ws.cap * sizeof(double));
            die_if_null(ws.buf, "ws M4 malloc failed");
            ws.top = 0;

            double *T2 = ws_alloc(&ws, kk);
            sub_view(B21, ldb, B11, ldb, T2, k, k);
            strassen_parallel(A22, lda, T2, k, M4, k, k, depth + 1);
            free(ws.buf);
        }

        #pragma omp task shared(M5)
        {
            Workspace ws;
            ws.cap = workspace_needed_doubles(k);
            ws.buf = (double*)malloc(ws.cap * sizeof(double));
            die_if_null(ws.buf, "ws M5 malloc failed");
            ws.top = 0;

            double *T1 = ws_alloc(&ws, kk);
            add_view(A11, lda, A12, lda, T1, k, k);
            strassen_parallel(T1, k, B22, ldb, M5, k, k, depth + 1);
            free(ws.buf);
        }

        #pragma omp task shared(M6)
        {
            Workspace ws;
            ws.cap = workspace_needed_doubles(k);
            ws.buf = (double*)malloc(ws.cap * sizeof(double));
            die_if_null(ws.buf, "ws M6 malloc failed");
            ws.top = 0;

            double *T1 = ws_alloc(&ws, kk);
            double *T2 = ws_alloc(&ws, kk);
            sub_view(A21, lda, A11, lda, T1, k, k);
            add_view(B11, ldb, B12, ldb, T2, k, k);
            strassen_parallel(T1, k, T2, k, M6, k, k, depth + 1);
            free(ws.buf);
        }

        #pragma omp task shared(M7)
        {
            Workspace ws;
            ws.cap = workspace_needed_doubles(k);
            ws.buf = (double*)malloc(ws.cap * sizeof(double));
            die_if_null(ws.buf, "ws M7 malloc failed");
            ws.top = 0;

            double *T1 = ws_alloc(&ws, kk);
            double *T2 = ws_alloc(&ws, kk);
            sub_view(A12, lda, A22, lda, T1, k, k);
            add_view(B21, ldb, B22, ldb, T2, k, k);
            strassen_parallel(T1, k, T2, k, M7, k, k, depth + 1);
            free(ws.buf);
        }
    }

    // zero C quadrants
    for (int i = 0; i < k; i++) {
        memset(C11 + i*ldc, 0, (size_t)k * sizeof(double));
        memset(C12 + i*ldc, 0, (size_t)k * sizeof(double));
        memset(C21 + i*ldc, 0, (size_t)k * sizeof(double));
        memset(C22 + i*ldc, 0, (size_t)k * sizeof(double));
    }

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

    free(block);
}

// Compute one Mi locally using OMP tasks inside the rank
static void compute_Mi_hybrid(int Mi_id,
                              const double *A, const double *B, int Npad,
                              double *Mi_out, int k) {
    const double *A11 = A;
    const double *A12 = A + k;
    const double *A21 = A + k * Npad;
    const double *A22 = A + k * Npad + k;

    const double *B11 = B;
    const double *B12 = B + k;
    const double *B21 = B + k * Npad;
    const double *B22 = B + k * Npad + k;

    // temps for building inputs (contiguous kxk)
    size_t kk = (size_t)k * (size_t)k;
    double *T1 = (double*)malloc(kk * sizeof(double));
    double *T2 = (double*)malloc(kk * sizeof(double));
    die_if_null(T1, "alloc T1 failed");
    die_if_null(T2, "alloc T2 failed");

    // Build the two operands as contiguous kxk (ld=k), then run strassen_parallel on them
    // Note: add_view/sub_view can write into contiguous buffers (ld=k)
    const int ld = k;

    switch (Mi_id) {
        case 1: // (A11+A22)(B11+B22)
            add_view(A11, Npad, A22, Npad, T1, ld, k);
            add_view(B11, Npad, B22, Npad, T2, ld, k);
            break;
        case 2: // (A21+A22)B11
            add_view(A21, Npad, A22, Npad, T1, ld, k);
            // copy B11 into T2
            for (int i = 0; i < k; i++) memcpy(T2 + (size_t)i*ld, B11 + (size_t)i*Npad, (size_t)k*sizeof(double));
            break;
        case 3: // A11(B12-B22)
            // copy A11 into T1
            for (int i = 0; i < k; i++) memcpy(T1 + (size_t)i*ld, A11 + (size_t)i*Npad, (size_t)k*sizeof(double));
            sub_view(B12, Npad, B22, Npad, T2, ld, k);
            break;
        case 4: // A22(B21-B11)
            // copy A22 into T1
            for (int i = 0; i < k; i++) memcpy(T1 + (size_t)i*ld, A22 + (size_t)i*Npad, (size_t)k*sizeof(double));
            sub_view(B21, Npad, B11, Npad, T2, ld, k);
            break;
        case 5: // (A11+A12)B22
            add_view(A11, Npad, A12, Npad, T1, ld, k);
            // copy B22 into T2
            for (int i = 0; i < k; i++) memcpy(T2 + (size_t)i*ld, B22 + (size_t)i*Npad, (size_t)k*sizeof(double));
            break;
        case 6: // (A21-A11)(B11+B12)
            sub_view(A21, Npad, A11, Npad, T1, ld, k);
            add_view(B11, Npad, B12, Npad, T2, ld, k);
            break;
        case 7: // (A12-A22)(B21+B22)
            sub_view(A12, Npad, A22, Npad, T1, ld, k);
            add_view(B21, Npad, B22, Npad, T2, ld, k);
            break;
        default:
            fprintf(stderr, "Invalid Mi_id=%d\n", Mi_id);
            MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Run OMP-task Strassen on contiguous operands
    #pragma omp parallel
    {
        #pragma omp single
        {
            strassen_parallel(T1, ld, T2, ld, Mi_out, ld, k, 0);
        }
    }

    free(T1);
    free(T2);
}

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // CLI: ./strassen_ompi [expected_N] [input_file]
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
            fprintf(stderr, "Strassen Hybrid: N=%d pads to %d (too large). Skipping.\n", N, Npad);
        }
        if (rank == 0) { free(A_in); free(B_in); }
        MPI_Finalize();
        return 2;
    }

    if (rank == 0 && Npad != N) {
        printf("Note: padding N=%d up to Npad=%d for Hybrid Strassen.\n", N, Npad);
    }

    size_t NN = (size_t)Npad * (size_t)Npad;
    double *A = (double*)calloc(NN, sizeof(double));
    double *B = (double*)calloc(NN, sizeof(double));
    double *C = NULL;
    die_if_null(A, "alloc A failed");
    die_if_null(B, "alloc B failed");

    if (rank == 0) {
        for (int i = 0; i < N; i++) {
            memcpy(A + (size_t)i * Npad, A_in + (size_t)i * N, (size_t)N * sizeof(double));
            memcpy(B + (size_t)i * Npad, B_in + (size_t)i * N, (size_t)N * sizeof(double));
        }
        free(A_in); free(B_in);

        C = (double*)calloc(NN, sizeof(double));
        die_if_null(C, "alloc C failed");
    }

    // broadcast padded A,B (simple, robust)
    MPI_Bcast(A, (int)NN, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(B, (int)NN, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    int k = Npad / 2;
    size_t kk = (size_t)k * (size_t)k;

    double *Mlocal = (double*)calloc(kk, sizeof(double));
    die_if_null(Mlocal, "alloc Mlocal failed");

    double *Mroot = NULL;
    if (rank == 0) {
        Mroot = (double*)malloc(kk * sizeof(double));
        die_if_null(Mroot, "alloc Mroot failed");
    }

    double *M1=NULL,*M2=NULL,*M3=NULL,*M4=NULL,*M5=NULL,*M6=NULL,*M7=NULL;
    if (rank == 0) {
        M1=(double*)malloc(kk*sizeof(double));
        M2=(double*)malloc(kk*sizeof(double));
        M3=(double*)malloc(kk*sizeof(double));
        M4=(double*)malloc(kk*sizeof(double));
        M5=(double*)malloc(kk*sizeof(double));
        M6=(double*)malloc(kk*sizeof(double));
        M7=(double*)malloc(kk*sizeof(double));
        die_if_null(M1,"alloc M1"); die_if_null(M2,"alloc M2"); die_if_null(M3,"alloc M3");
        die_if_null(M4,"alloc M4"); die_if_null(M5,"alloc M5"); die_if_null(M6,"alloc M6"); die_if_null(M7,"alloc M7");
    }

    // distribute Mi round-robin across ranks; each rank uses OpenMP internally
    for (int Mi_id = 1; Mi_id <= 7; Mi_id++) {
        memset(Mlocal, 0, kk * sizeof(double));

        if (((Mi_id - 1) % size) == rank) {
            compute_Mi_hybrid(Mi_id, A, B, Npad, Mlocal, k);
        }

        MPI_Reduce(Mlocal, Mroot, (int)kk, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            switch (Mi_id) {
                case 1: memcpy(M1, Mroot, kk*sizeof(double)); break;
                case 2: memcpy(M2, Mroot, kk*sizeof(double)); break;
                case 3: memcpy(M3, Mroot, kk*sizeof(double)); break;
                case 4: memcpy(M4, Mroot, kk*sizeof(double)); break;
                case 5: memcpy(M5, Mroot, kk*sizeof(double)); break;
                case 6: memcpy(M6, Mroot, kk*sizeof(double)); break;
                case 7: memcpy(M7, Mroot, kk*sizeof(double)); break;
            }
        }
    }

    if (rank == 0) {
        double *C11 = C;
        double *C12 = C + k;
        double *C21 = C + k * Npad;
        double *C22 = C + k * Npad + k;

        // zero quadrants
        for (int i = 0; i < k; i++) {
            memset(C11 + i*Npad, 0, (size_t)k*sizeof(double));
            memset(C12 + i*Npad, 0, (size_t)k*sizeof(double));
            memset(C21 + i*Npad, 0, (size_t)k*sizeof(double));
            memset(C22 + i*Npad, 0, (size_t)k*sizeof(double));
        }

        // combine
        axpy_view(+1.0, M1, k, C11, Npad, k);
        axpy_view(+1.0, M4, k, C11, Npad, k);
        axpy_view(-1.0, M5, k, C11, Npad, k);
        axpy_view(+1.0, M7, k, C11, Npad, k);

        axpy_view(+1.0, M3, k, C12, Npad, k);
        axpy_view(+1.0, M5, k, C12, Npad, k);

        axpy_view(+1.0, M2, k, C21, Npad, k);
        axpy_view(+1.0, M4, k, C21, Npad, k);

        axpy_view(+1.0, M1, k, C22, Npad, k);
        axpy_view(-1.0, M2, k, C22, Npad, k);
        axpy_view(+1.0, M3, k, C22, Npad, k);
        axpy_view(+1.0, M6, k, C22, Npad, k);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    if (rank == 0) {
        printf("Hybrid Strassen (MPI top-level + OMP inside) (N=%d, padded=%d, np=%d, threads=%d) Time: %f s\n",
               N, Npad, size, omp_get_max_threads(), t1 - t0);
               verify_or_warn("[strassen_omp]", A, B, C, N, Npad);

    }

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
