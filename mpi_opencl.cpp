#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <CL/cl.h>

#define MATRIX_SIZE 20
#define WORK_GROUP_SIZE 16

void matrixMultiplyOpenCL(int *a, int *b, int *c, int size, cl_device_id device_id, const char *source);

char *readKernelFile(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Failed to open %s\n", filename);
        exit(EXIT_FAILURE);
    }
    fseek(file, 0, SEEK_END);
    long length = ftell(file);
    fseek(file, 0, SEEK_SET);
    char *source = (char *)malloc(length + 1);
    fread(source, 1, length, file);
    fclose(file);
    source[length] = '\0';
    return source;
}

int main(int argc, char *argv[]) {
    int rank, size;
    int a[MATRIX_SIZE][MATRIX_SIZE], b[MATRIX_SIZE][MATRIX_SIZE], c[MATRIX_SIZE][MATRIX_SIZE];
    double start_time, end_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        start_time = MPI_Wtime();
    }

    if (rank == 0) {
        for (int i = 0; i < MATRIX_SIZE; i++) {
            for (int j = 0; j < MATRIX_SIZE; j++) {
                a[i][j] = i + j;
                b[i][j] = i - j;
            }
        }
    }

    MPI_Bcast(&a, MATRIX_SIZE * MATRIX_SIZE, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&b, MATRIX_SIZE * MATRIX_SIZE, MPI_INT, 0, MPI_COMM_WORLD);

    char *source = readKernelFile("ops.cl");

    matrixMultiplyOpenCL((int *)a, (int *)b, (int *)c, MATRIX_SIZE, NULL, source);

    if (rank == 0) {
        end_time = MPI_Wtime();
        printf("Matrix multiplication result:\n");
        for (int i = 0; i < MATRIX_SIZE; i++) {
            for (int j = 0; j < MATRIX_SIZE; j++) {
                printf("%d ", c[i][j]);
            }
            printf("\n");
        }
        printf("Execution time: %f seconds\n", end_time - start_time);
    }

    MPI_Finalize();
    return 0;
}

void matrixMultiplyOpenCL(int *a, int *b, int *c, int size, cl_device_id device_id, const char *source) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            c[i * size + j] = a[i * size + j] * b[i * size + j];
        }
    }
}
