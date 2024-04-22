#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <mpi.h>
#define m_Matrix 50

void rMatrix(int N, int **m) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            m[i][j] = rand() % 10; // Generate random values between 0 and 9
        }
    }
}

void mMatrix(int N, int **A, int **B, int **C) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0;
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void ToFile(int N, int **mat, const char *filename) {
    FILE *f = fopen(filename, "w");
    if (f == NULL) {
        printf("Error opening file.\n");
        return;
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            fprintf(f, "%d ", mat[i][j]);
        }
        fprintf(f, "\n");
    }
    fclose(f);
}

int main(int argc, char **argv) {
    srand(time(NULL)); // Seed the random number generator
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int N =  m_Matrix;
    printf("Matrix Size: %d x %d\n", N, N);

    // Allocate memory for matrices A, B, and C
    int **A = (int **)malloc(N * sizeof(int *));
    int **B = (int **)malloc(N * sizeof(int *));
    int **C = (int **)malloc(N * sizeof(int *));
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
    
    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    
    // Divide work among processes
    int c_size = N / size;
    int starting_row = rank * c_size;
    int ending_row = (rank == size - 1) ? N : (rank + 1) * c_size;
    
    // Multiply matrices A and B
    mMatrix(N, A, B, C);
    
    // Gather results from all processes to process 0
    MPI_Gather(C[starting_row], c_size * N, MPI_INT, C[0], c_size * N, MPI_INT, 0, MPI_COMM_WORLD);
    
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    double exe_time = (end_time.tv_sec - start_time.tv_sec) * 1e6 +
                      (end_time.tv_nsec - start_time.tv_nsec) / 1e3;
    
    // Print execution time on process 0
    if (rank == 0) {
        printf("Execution time: %.6f microseconds\n", exe_time);
    }

    // Free dynamically allocated memory
    for (int i = 0; i < N; i++) {
        free(A[i]);
        free(B[i]);
        free(C[i]);
    }
    free(A);
    free(B);
    free(C);
    
    MPI_Finalize();
    return 0;
}
