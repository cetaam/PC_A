import random
import sys

def generate_matrix_file(N, filename="matrix_input.txt"):
    print(f"Generating {N}x{N} matrices into {filename}...")
    
    with open(filename, 'w') as f:
        # Write Size
        f.write(f"{N}\n")
        
        # Write Matrix A (Row-major)
        # Using integers for cleaner checking, change to random.uniform for doubles
        for i in range(N * N):
            f.write(f"{random.randint(1, 10)} ")
        f.write("\n")
        
        # Write Matrix B
        for i in range(N * N):
            f.write(f"{random.randint(1, 10)} ")
        f.write("\n")
        
    print("Done!")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        N = int(sys.argv[1])
    else:
        N = 64 # Default size
    
    generate_matrix_file(N)