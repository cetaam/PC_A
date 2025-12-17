# Compiler definitions
CXX    = g++
MPICXX = mpic++

# Flags
# -fopenmp is required for both OMP and the Hybrid target
CXXFLAGS     = -fopenmp -O2 -Wall -I. -DUSE_BLAS -DVERIFY_FULL_MAX=1024 -DVERIFY_SPOT_SAMPLES=64
MPIFLAGS     = -O2 -Wall -I. -DUSE_BLAS -DVERIFY_FULL_MAX=1024 -DVERIFY_SPOT_SAMPLES=64
HYBRIDFLAGS  = -fopenmp -O2 -Wall -I. -DUSE_BLAS -DVERIFY_FULL_MAX=1024 -DVERIFY_SPOT_SAMPLES=64
BLASLIBS = -lopenblas


# Matrix Size (Default 1024, override with "make run_naive N=2048")
N       ?= 1024
PROCS   ?= 4
THREADS ?= 4
TOTAL ?= $(shell echo $$(( $(PROCS) * $(THREADS) )))

# Allow overriding the mpirun command (some clusters use srun, mpiexec, etc.)
MPIRUN ?= mpirun
HOSTFILE ?= host.txt

MPI_HOSTFLAGS = --hostfile $(HOSTFILE)
# MPI-only: one rank per node is typical
MPI_RUN = $(MPIRUN) $(MPI_HOSTFLAGS) -np $(PROCS)
# Target Binaries
TARGETS = naive_mpi naive_ompi strassen_mpi strassen_ompi

# Default: Build all naive binaries
all: $(TARGETS)

# --- Compilation Rules ---

naive_omp: naive_omp.cpp matrix_io.h
	$(CXX) $(CXXFLAGS) naive_omp.cpp -o naive_omp $(BLASLIBS)

naive_mpi: naive_mpi.cpp matrix_io.h
	$(MPICXX) $(MPIFLAGS) naive_mpi.cpp -o naive_mpi $(BLASLIBS)

naive_ompi: naive_ompi.cpp matrix_io.h
	$(MPICXX) $(HYBRIDFLAGS) naive_ompi.cpp -o naive_ompi $(BLASLIBS)

strassen_omp: strassen_omp.cpp matrix_io.h
	$(CXX) $(CXXFLAGS) strassen_omp.cpp -o strassen_omp $(BLASLIBS)

strassen_mpi: strassen_mpi.cpp matrix_io.h
	$(MPICXX) $(MPIFLAGS) strassen_mpi.cpp -o strassen_mpi $(BLASLIBS)

strassen_ompi: strassen_ompi.cpp matrix_io.h
	$(MPICXX) $(HYBRIDFLAGS) strassen_ompi.cpp -o strassen_ompi $(BLASLIBS)

# --- Helper Rules ---

# Generate Matrix Data
generate:
	python3 gen_matrix.py $(N)

# Clean up binaries and text files
clean:
	rm -f $(TARGETS) matrix_input.txt

# --- The "One-Click" Run Command ---
# Usage: make run_naive
# Usage: make run_naive N=2000 PROCS=4 THREADS=8
run_naive: clean generate all

	@echo "=========================================================="
	@echo "Running Evaluation for Matrix Size N=$(N)"
	@echo "PROCS=$(PROCS), THREADS=$(THREADS), TOTAL=$(TOTAL)"
	@echo "=========================================================="

	@echo "\n>>> 1. Running naive OpenMP (Shared Memory) with TOTAL=$(THREADS) threads..."
	OMP_NUM_THREADS=$(THREADS) ./naive_omp $(N) matrix_input.txt

	@echo "\n>>>1. Running Strassen OpenMP (Shared Memory) with TOTAL=$(THREADS) threads..."
	OMP_NUM_THREADS=$(THREADS) ./strassen_omp $(N) matrix_input.txt

#	@echo "\n>>> 2. Running MPI (Distributed Memory) with TOTAL=$(PROCS) processes..."
#	$(MPIRUN) --allow-run-as-root --bind-to hwthread --map-by hwthread  -np $(PROCS) ./naive_mpi $(N) matrix_input.txt
	
#	@echo "\n>>>2. Running Strassen MPI (Distributed Memory) with TOTAL=$(PROCS) processes..."
#	$(MPIRUN) --allow-run-as-root --bind-to hwthread --map-by hwthread  -np $(PROCS) ./strassen_mpi $(N) matrix_input.txt
#	@echo "\n>>> 3. Running Hybrid (MPI + OpenMP)..."
#	@echo "    (Using $(PROCS) MPI Processes, each with $(THREADS) Threads)"
#	OMP_NUM_THREADS=$(THREADS) $(MPIRUN) --allow-run-as-root --bind-to hwthread --map-by hwthread  -np $(PROCS) ./naive_ompi $(N) matrix_input.txt

#	@echo "\n>>>3. Running Strassen Hybrid (MPI + OpenMP)..."
#	@echo "    (Using $(PROCS) MPI Processes, each with $(THREADS) Threads)"
#	OMP_NUM_THREADS=$(THREADS) $(MPIRUN) --allow-run-as-root --bind-to hwthread --map-by hwthread  -np $(PROCS) ./strassen_ompi $(N) matrix_input.txt
	@echo "\n=========================================================="
	@echo "Done."

# --- The "Cluster" Run Command ---
run_cluster: generate all
	@echo "=========================================================="
	@echo "CLUSTER RUN N=$(N) hostfile=$(HOSTFILE) PROCS=$(PROCS) THREADS=$(THREADS)"
	@echo "=========================================================="

#	@echo "\n>>> 1) OpenMP (single node only) THREADS=$(THREADS)"
#	OMP_NUM_THREADS=$(THREADS) ./naive_omp $(N) matrix_input.txt

	@echo "\n>>> 2) MPI naive: PROCS=$(PROCS)"
	$(MPI_RUN) ./naive_mpi $(N) matrix_input.txt

	@echo "\n>>> 3) Hybrid naive: PROCS=$(PROCS) x THREADS=$(THREADS)"
	OMP_NUM_THREADS=$(THREADS) $(HYBRID_RUN) ./naive_ompi $(N) matrix_input.txt

	@echo "\n>>> 4) MPI Strassen"
	$(MPI_RUN) ./strassen_mpi $(N) matrix_input.txt

	@echo "\n>>> 5) Hybrid Strassen"
	OMP_NUM_THREADS=$(THREADS) $(HYBRID_RUN) ./strassen_ompi $(N) matrix_input.txt



.PHONY: all clean generate run_naive
