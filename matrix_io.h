#ifndef MATRIX_IO_H
#define MATRIX_IO_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef USE_BLAS
  #include <cblas.h>
#endif

#ifndef VERIFY_FULL_MAX
#define VERIFY_FULL_MAX 4096    // full BLAS verify up to this N
#endif

#ifndef VERIFY_SPOT_SAMPLES
#define VERIFY_SPOT_SAMPLES 256 // spot-check samples for large N
#endif

#ifndef VERIFY_SPOT_TOL
#define VERIFY_SPOT_TOL 1e-6    // absolute tolerance for spotcheck (double)
#endif

/*
File format (row-major):
  N
  A[0..N*N-1]
  B[0..N*N-1]
*/

static inline int read_matrix_file(const char *filename, int *N, double **A, double **B) {
    FILE *f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "Error: Could not open file '%s'\n", filename);
        return -1;
    }

    if (fscanf(f, "%d", N) != 1 || *N <= 0) {
        fprintf(stderr, "Error: Failed to read valid N from '%s'\n", filename);
        fclose(f);
        return -1;
    }

    size_t total = (size_t)(*N) * (size_t)(*N);
    *A = (double*)malloc(total * sizeof(double));
    *B = (double*)malloc(total * sizeof(double));
    if (!*A || !*B) {
        fprintf(stderr, "Error: malloc failed for A/B (N=%d)\n", *N);
        free(*A); free(*B);
        fclose(f);
        return -1;
    }

    for (size_t i = 0; i < total; i++) {
        if (fscanf(f, "%lf", &(*A)[i]) != 1) {
            fprintf(stderr, "Error: Failed reading A at index %zu\n", i);
            free(*A); free(*B);
            fclose(f);
            return -1;
        }
    }

    for (size_t i = 0; i < total; i++) {
        if (fscanf(f, "%lf", &(*B)[i]) != 1) {
            fprintf(stderr, "Error: Failed reading B at index %zu\n", i);
            free(*A); free(*B);
            fclose(f);
            return -1;
        }
    }

    fclose(f);
    return 0;
}

static inline void save_result(const char *filename, const double *C, int N) {
    FILE *f = fopen(filename, "w");
    if (!f) {
        fprintf(stderr, "Error: Could not open '%s' for writing\n", filename);
        return;
    }
    fprintf(f, "%d\n", N);
    for (int i = 0; i < N*N; i++) fprintf(f, "%.12g ", C[i]);
    fprintf(f, "\n");
    fclose(f);
}

#ifdef USE_BLAS
// Verify top-left N×N of C. A/B/C may be padded with leading dimension ld (>= N).
static inline int verify_with_blas_top_left(const double *A, const double *B,
                                            const double *C, int N, int Nc,
                                            double rtol, double atol,
                                            double *out_max_abs, double *out_max_rel) {
    // IMPORTANT: if C is padded (Nc > N), Strassen also uses padded A/B.
    // We assume A and B have the same leading dimension as C when Nc > N.
    const int ldAB = (Nc >= N ? Nc : N);

    size_t total = (size_t)N * (size_t)N;
    double *Cref = (double*)calloc(total, sizeof(double));
    if (!Cref) return 2;

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N, N, N,
                1.0, A, ldAB,
                     B, ldAB,
                0.0, Cref, N);

    double max_abs = 0.0, max_rel = 0.0;

    for (int i = 0; i < N; i++) {
        const double *crow = C    + (size_t)i * (size_t)Nc;
        const double *rrow = Cref + (size_t)i * (size_t)N;
        for (int j = 0; j < N; j++) {
            double diff = fabs(crow[j] - rrow[j]);
            double denom = fabs(rrow[j]) + 1e-12;
            double rel = diff / denom;
            if (diff > max_abs) max_abs = diff;
            if (rel  > max_rel) max_rel = rel;

            if (diff > atol && rel > rtol) {
                free(Cref);
                if (out_max_abs) *out_max_abs = max_abs;
                if (out_max_rel) *out_max_rel = max_rel;
                return 1;
            }
        }
    }

    free(Cref);
    if (out_max_abs) *out_max_abs = max_abs;
    if (out_max_rel) *out_max_rel = max_rel;
    return 0;
}
#endif

// Fast correctness check for large N: recompute a few random entries of C and compare.
// Uses top-left N×N of C, with leading dimension Nc.
static inline int verify_spotcheck_top_left(const double *A, const double *B,
                                            const double *C, int N, int Nc,
                                            int samples, double tol,
                                            double *out_max_abs) {
    // Same stride rule as BLAS verify: if C is padded (Nc>N), A/B are padded too.
    const int ldAB = (Nc >= N ? Nc : N);

    unsigned int seed = 12345u;
    double max_abs = 0.0;

    for (int s = 0; s < samples; s++) {
        int i = (int)(rand_r(&seed) % (unsigned)N);
        int j = (int)(rand_r(&seed) % (unsigned)N);

        double sum = 0.0;
        const double *arow = A + (size_t)i * (size_t)ldAB;
        for (int k = 0; k < N; k++) {
            sum += arow[k] * B[(size_t)k * (size_t)ldAB + (size_t)j];
        }

        double diff = fabs(C[(size_t)i * (size_t)Nc + (size_t)j] - sum);
        if (diff > max_abs) max_abs = diff;

        if (diff > tol) {
            if (out_max_abs) *out_max_abs = max_abs;
            return 1;
        }
    }

    if (out_max_abs) *out_max_abs = max_abs;
    return 0;
}

// Call this after computing C (every run).
// - For N <= VERIFY_FULL_MAX: full BLAS verification (strong, slower)
// - For N  > VERIFY_FULL_MAX: spot-check verification (fast)
static inline void verify_or_warn(const char *tag,
                                  const double *A, const double *B,
                                  const double *C,
                                  int N, int Nc) {
#ifdef USE_BLAS
    if (N <= VERIFY_FULL_MAX) {
        double max_abs=0.0, max_rel=0.0;
        int ok = verify_with_blas_top_left(A, B, C, N, Nc, 1e-8, 1e-8, &max_abs, &max_rel);
        if (ok == 0) {
            printf("%s VERIFY OK (BLAS): max_abs=%e max_rel=%e\n", tag, max_abs, max_rel);
        } else {
            printf("%s VERIFY FAIL (BLAS): max_abs=%e max_rel=%e\n", tag, max_abs, max_rel);
        }
    } else {
        double max_abs=0.0;
        int ok = verify_spotcheck_top_left(A, B, C, N, Nc, VERIFY_SPOT_SAMPLES, VERIFY_SPOT_TOL, &max_abs);
        if (ok == 0) {
            printf("%s VERIFY OK (spotcheck %d): max_abs=%e (tol=%e)\n",
                   tag, VERIFY_SPOT_SAMPLES, max_abs, (double)VERIFY_SPOT_TOL);
        } else {
            printf("%s VERIFY FAIL (spotcheck %d): max_abs=%e (tol=%e)\n",
                   tag, VERIFY_SPOT_SAMPLES, max_abs, (double)VERIFY_SPOT_TOL);
        }
    }
#else
    (void)tag; (void)A; (void)B; (void)C; (void)N; (void)Nc;
    printf("%s VERIFY SKIPPED (compile with -DUSE_BLAS and link BLAS)\n", tag);
#endif
}

#endif // MATRIX_IO_H
