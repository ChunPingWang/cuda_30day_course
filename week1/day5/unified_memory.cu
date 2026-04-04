#include <stdio.h>

/**
 * Using Unified Memory - Simplified program
 *
 * Advantages:
 * - No need to manually allocate two copies of memory
 * - No need to manually copy data
 * - Cleaner code
 *
 * Disadvantages:
 * - Performance may be slightly lower (automatic transfer overhead)
 * - Requires CUDA 6.0+
 */

__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    printf("========================================\n");
    printf("    Unified Memory Demo\n");
    printf("========================================\n\n");

    const int n = 10;
    size_t bytes = n * sizeof(float);

    // ========== Allocate unified memory using cudaMallocManaged ==========
    printf("1. Allocate unified memory using cudaMallocManaged\n");

    float *a, *b, *c;
    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytes);

    printf("   Allocated %zu bytes x 3 of unified memory\n\n", bytes);

    // ========== Initialize directly on CPU (no cudaMemcpy needed!) ==========
    printf("2. Initialize data directly on CPU\n");

    printf("   A = [ ");
    for (int i = 0; i < n; i++) {
        a[i] = (float)i;
        printf("%.0f ", a[i]);
    }
    printf("]\n");

    printf("   B = [ ");
    for (int i = 0; i < n; i++) {
        b[i] = (float)(i * 10);
        printf("%.0f ", b[i]);
    }
    printf("]\n\n");

    // ========== Execute Kernel ==========
    printf("3. Execute GPU kernel\n");

    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    vectorAdd<<<blocks, threadsPerBlock>>>(a, b, c, n);

    // Important: Wait for GPU to complete
    cudaDeviceSynchronize();

    printf("   Kernel execution complete!\n\n");

    // ========== Read results directly on CPU (no cudaMemcpy needed!) ==========
    printf("4. Read results directly on CPU\n");
    printf("   C = A + B = [ ");
    for (int i = 0; i < n; i++) {
        printf("%.0f ", c[i]);
    }
    printf("]\n\n");

    // Verify results
    printf("5. Verify results\n");
    bool correct = true;
    for (int i = 0; i < n; i++) {
        if (c[i] != a[i] + b[i]) {
            printf("    Error: c[%d] = %.0f, expected %.0f\n", i, c[i], a[i] + b[i]);
            correct = false;
        }
    }
    if (correct) {
        printf("    All results correct!\n\n");
    }

    // ========== Free Memory ==========
    printf("6. Free memory\n");
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    printf("   Memory freed\n");

    // ========== Code Comparison ==========
    printf("\n========================================\n");
    printf("Code Comparison:\n");
    printf("========================================\n\n");

    printf("Traditional approach requires:\n");
    printf("  - malloc() and cudaMalloc() for separate allocation\n");
    printf("  - cudaMemcpy(..., HostToDevice) to send data\n");
    printf("  - cudaMemcpy(..., DeviceToHost) to get results\n");
    printf("  - free() and cudaFree() for separate freeing\n\n");

    printf("Unified memory only needs:\n");
    printf("  - cudaMallocManaged() for single allocation\n");
    printf("  - cudaDeviceSynchronize() to ensure completion\n");
    printf("  - cudaFree() for single freeing\n");

    printf("\n========================================\n");
    printf(" Unified memory demo complete!\n");
    printf("========================================\n");

    return 0;
}
