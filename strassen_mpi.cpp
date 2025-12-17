#include <mpi.h>
#include <iostream>
#include <vector>

using namespace std;

const int N = 4; // Size of the matrix (NxN)

void addMatrix(const vector<vector<int>> &A, const vector<vector<int>> &B, vector<vector<int>> &result, int size) {
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            result[i][j] = A[i][j] + B[i][j];
}

void subtractMatrix(const vector<vector<int>> &A, const vector<vector<int>> &B, vector<vector<int>> &result, int size) {
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            result[i][j] = A[i][j] - B[i][j];
}

void strassen(const vector<vector<int>> &A, const vector<vector<int>> &B, vector<vector<int>> &C, int size) {
    if (size == 1) {
        C[0][0] = A[0][0] * B[0][0];
        return;
    }

    int newSize = size / 2;
    vector<vector<int>> a11(newSize, vector<int>(newSize));
    vector<vector<int>> a12(newSize, vector<int>(newSize));
    vector<vector<int>> a21(newSize, vector<int>(newSize));
    vector<vector<int>> a22(newSize, vector<int>(newSize));
    vector<vector<int>> b11(newSize, vector<int>(newSize));
    vector<vector<int>> b12(newSize, vector<int>(newSize));
    vector<vector<int>> b21(newSize, vector<int>(newSize));
    vector<vector<int>> b22(newSize, vector<int>(newSize));
    vector<vector<int>> c11(newSize, vector<int>(newSize));
    vector<vector<int>> c12(newSize, vector<int>(newSize));
    vector<vector<int>> c21(newSize, vector<int>(newSize));
    vector<vector<int>> c22(newSize, vector<int>(newSize));
    vector<vector<int>> p1(newSize, vector<int>(newSize));
    vector<vector<int>> p2(newSize, vector<int>(newSize));
    vector<vector<int>> p3(newSize, vector<int>(newSize));
    vector<vector<int>> p4(newSize, vector<int>(newSize));
    vector<vector<int>> p5(newSize, vector<int>(newSize));
    vector<vector<int>> p6(newSize, vector<int>(newSize));
    vector<vector<int>> p7(newSize, vector<int>(newSize));
    vector<vector<int>> aResult(newSize, vector<int>(newSize));
    vector<vector<int>> bResult(newSize, vector<int>(newSize));

    for (int i = 0; i < newSize; i++) {
        for (int j = 0; j < newSize; j++) {
            a11[i][j] = A[i][j];
            a12[i][j] = A[i][j + newSize];
            a21[i][j] = A[i + newSize][j];
            a22[i][j] = A[i + newSize][j + newSize];
            b11[i][j] = B[i][j];
            b12[i][j] = B[i][j + newSize];
            b21[i][j] = B[i + newSize][j];
            b22[i][j] = B[i + newSize][j + newSize];
        }
    }

    addMatrix(a11, a22, aResult, newSize);
    addMatrix(b11, b22, bResult, newSize);
    strassen(aResult, bResult, p1, newSize);

    addMatrix(a21, a22, aResult, newSize);
    strassen(aResult, b11, p2, newSize);

    subtractMatrix(b12, b22, bResult, newSize);
    strassen(a11, bResult, p3, newSize);

    subtractMatrix(b21, b11, bResult, newSize);
    strassen(a22, bResult, p4, newSize);

    addMatrix(a11, a12, aResult, newSize);
    strassen(aResult, b22, p5, newSize);

    subtractMatrix(a21, a11, aResult, newSize);
    addMatrix(b11, b12, bResult, newSize);
    strassen(aResult, bResult, p6, newSize);

    subtractMatrix(a12, a22, aResult, newSize);
    addMatrix(b21, b22, bResult, newSize);
    strassen(aResult, bResult, p7, newSize);

    addMatrix(p1, p4, aResult, newSize);
    subtractMatrix(aResult, p5, bResult, newSize);
    addMatrix(bResult, p7, c11, newSize);

    addMatrix(p3, p5, c12, newSize);

    addMatrix(p2, p4, c21, newSize);

    addMatrix(p1, p3, aResult, newSize);
    subtractMatrix(aResult, p2, bResult, newSize);
    addMatrix(bResult, p6, c22, newSize);

    for (int i = 0; i < newSize; i++) {
        for (int j = 0; j < newSize; j++) {
            C[i][j] = c11[i][j];
            C[i][j + newSize] = c12[i][j];
            C[i + newSize][j] = c21[i][j];
            C[i + newSize][j + newSize] = c22[i][j];
        }
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    vector<vector<int>> A = {{1, 2, 3, 4},
                             {5, 6, 7, 8},
                             {9, 10, 11, 12},
                             {13, 14, 15, 16}};

    vector<vector<int>> B = {{16, 15, 14, 13},
                             {12, 11, 10, 9},
                             {8, 7, 6, 5},
                             {4, 3, 2, 1}};

    vector<vector<int>> C(N, vector<int>(N, 0));

    strassen(A, B, C, N);

    // Print result
    for (auto &row : C) {
        for (auto &elem : row) {
            cout << elem << " ";
        }
        cout << endl;
    }

    MPI_Finalize();

    return 0;
}