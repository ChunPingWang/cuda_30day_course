#include <stdio.h>
#include <math.h>

/**
 * Exercise 3: Matrix Scaling
 * Multiply each element in the matrix by constant k
 */

__global__ void matrixScale(float *matrix, float k, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        matrix[idx] = matrix[idx] * k;
    }
}

void printMatrix(float *matrix, int width, int height, const char *name) {
    printf("%s:\n", name);
    for (int y = 0; y < height; y++) {
        printf("  [ ");
        for (int x = 0; x < width; x++) {
            printf("%5.1f ", matrix[y * width + x]);
        }
        printf("]\n");
    }
    printf("\n");
}

int main() {
    printf("========================================\n");
    printf("    Exercise 3: Matrix Scaling\n");
    printf("========================================\n\n");

    const int width = 4;
    const int height = 3;
    const float k = 2.5f;
    size_t bytes = width * height * sizeof(float);

    // Use unified memory
    float *matrix;
    cudaMallocManaged(&matrix, bytes);

    // Initialize matrix
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            matrix[y * width + x] = (float)(y * width + x + 1);
        }
    }

    printMatrix(matrix, width, height, "Original Matrix");
    printf("Scale factor k = %.1f\n\n", k);

    // Execute kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + 15) / 16, (height + 15) / 16);

    matrixScale<<<numBlocks, threadsPerBlock>>>(matrix, k, width, height);
    cudaDeviceSynchronize();

    printMatrix(matrix, width, height, "Scaled Matrix (x2.5)");

    // Verify
    bool correct = true;
    for (int i = 0; i < width * height; i++) {
        float expected = (i + 1) * k;
        if (fabs(matrix[i] - expected) > 0.001f) {
            correct = false;
            break;
        }
    }
    printf("Result verification: %s\n", correct ? "CORRECT" : "ERROR");

    // Free memory
    cudaFree(matrix);

    return 0;
}
