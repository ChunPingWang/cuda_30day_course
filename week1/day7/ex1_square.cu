#include <stdio.h>

/**
 * Exercise 1: Vector Square
 * Compute B[i] = A[i] * A[i]
 */

__global__ void vectorSquare(float *a, float *b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        b[idx] = a[idx] * a[idx];
    }
}

int main() {
    printf("========================================\n");
    printf("    Exercise 1: Vector Square\n");
    printf("========================================\n\n");

    const int n = 10;
    size_t bytes = n * sizeof(float);

    // Use unified memory
    float *a, *b;
    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);

    // Initialize
    printf("A = [ ");
    for (int i = 0; i < n; i++) {
        a[i] = (float)(i + 1);
        printf("%.0f ", a[i]);
    }
    printf("]\n\n");

    // Execute kernel
    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    vectorSquare<<<blocks, threadsPerBlock>>>(a, b, n);
    cudaDeviceSynchronize();

    // Display results
    printf("B = A^2 = [ ");
    for (int i = 0; i < n; i++) {
        printf("%.0f ", b[i]);
    }
    printf("]\n\n");

    // Verify
    bool correct = true;
    for (int i = 0; i < n; i++) {
        if (b[i] != a[i] * a[i]) {
            correct = false;
            break;
        }
    }
    printf("Result verification: %s\n", correct ? "CORRECT" : "ERROR");

    // Free memory
    cudaFree(a);
    cudaFree(b);

    return 0;
}
