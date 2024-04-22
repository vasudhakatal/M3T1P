#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

#define  m_Matrix 100

void rMatrix(int N, int **m) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            m[i][j] = rand() % 10;
        }
    }
}

void mMatrix(int N, int **A, int **B, int **C) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0;
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

int main(int argc, char **argv) {
    srand(time(NULL)); // Seed the random number generator
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int N = m_Matrix; // Random matrix size between 1 and 90
    printf("Matrix Size: %d x %d\n", N, N);
    int **A, **B, **C;

    // Allocate memory for matrices A, B, and C
    A = (int **)malloc(N * sizeof(int *));
    B = (int **)malloc(N * sizeof(int *));
    C = (int **)malloc(N * sizeof(int *));
    for (int i = 0; i < N; i++) {
        A[i] = (int *)malloc(N * sizeof(int));
        B[i] = (int *)malloc(N * sizeof(int));
        C[i] = (int *)malloc(N * sizeof(int));
    }

    if (rank == 0) {
        // Generate random matrices A and B
        rMatrix(N, A);
        rMatrix(N, B);
    }

    // Broadcast matrices A and B to all processes
    MPI_Bcast(A[0], N * N, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(B[0], N * N, MPI_INT, 0, MPI_COMM_WORLD);

    // Divide work among processes
    int c_size = N / size;
    int s_row = rank * c_size;
    int e_row = (rank == size - 1) ? N : (rank + 1) * c_size;

    // Measure execution time
    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    // Multiply matrices A and B
    mMatrix(N, A, B, C);

    // Gather results from all processes to process 0
    MPI_Gather(C + s_row, c_size * N, MPI_INT, C, c_size * N, MPI_INT, 0, MPI_COMM_WORLD);

    // Measure end time
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    double exe_time = (end_time.tv_sec - start_time.tv_sec) * 1e6 +
                      (end_time.tv_nsec - start_time.tv_nsec) / 1e3;

    // Print execution time on process 0
    if (rank == 0) {
        printf("Execution time: %.6f microseconds\n", exe_time);
    }

    MPI_Finalize();

    // Free allocated memory
    for (int i = 0; i < N; i++) {
        free(A[i]);
        free(B[i]);
        free(C[i]);
    }
    free(A);
    free(B);
    free(C);

    return 0;
}