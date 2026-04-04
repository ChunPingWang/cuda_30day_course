#include <stdio.h>

/**
 * Exercise 2: Vector Dot Product
 * result = A[0]*B[0] + A[1]*B[1] + ... + A[n-1]*B[n-1]
 *
 * This is a simplified version:
 * - GPU computes element-wise products
 * - CPU performs final summation
 * (More efficient method will be learned in Week 3: Parallel Reduction)
 */

__global__ void elementwiseMultiply(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

int main() {
    printf("========================================\n");
    printf("    Exercise 2: Vector Dot Product\n");
    printf("========================================\n\n");

    const int n = 8;
    size_t bytes = n * sizeof(float);

    // Use unified memory
    float *a, *b, *c;
    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytes);

    // Initialize
    printf("A = [ ");
    for (int i = 0; i < n; i++) {
        a[i] = (float)(i + 1);
        printf("%.0f ", a[i]);
    }
    printf("]\n");

    printf("B = [ ");
    for (int i = 0; i < n; i++) {
        b[i] = (float)(i + 1);
        printf("%.0f ", b[i]);
    }
    printf("]\n\n");

    // GPU: Compute element-wise products
    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    elementwiseMultiply<<<blocks, threadsPerBlock>>>(a, b, c, n);
    cudaDeviceSynchronize();

    printf("A * B (element-wise) = [ ");
    for (int i = 0; i < n; i++) {
        printf("%.0f ", c[i]);
    }
    printf("]\n\n");

    // CPU: Sum up
    float dotProduct = 0.0f;
    for (int i = 0; i < n; i++) {
        dotProduct += c[i];
    }

    printf("Dot Product = ");
    for (int i = 0; i < n; i++) {
        printf("%.0f", a[i] * b[i]);
        if (i < n - 1) printf(" + ");
    }
    printf("\n");
    printf("           = %.0f\n\n", dotProduct);

    // Verify
    float expected = 0.0f;
    for (int i = 0; i < n; i++) {
        expected += (i + 1) * (i + 1);  // 1^2 + 2^2 + 3^2 + ...
    }
    printf("Expected result: %.0f\n", expected);
    printf("Result verification: %s\n", (dotProduct == expected) ? "CORRECT" : "ERROR");

    // Free memory
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    printf("\nNote: This version performs summation on CPU.\n");
    printf("In Week 3, we will learn how to efficiently perform\n");
    printf("reduction operations on GPU.\n");

    return 0;
}
