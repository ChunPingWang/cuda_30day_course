#include <stdio.h>

/**
 * Traditional approach: each thread processes one element
 */
__global__ void traditionalProcess(float *arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        arr[idx] = arr[idx] * 2.0f;
    }
}

/**
 * Grid-Stride Loop: each thread processes multiple elements
 */
__global__ void gridStrideProcess(float *arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;  // Total number of threads

    // Each thread processes multiple elements with stride
    for (int i = idx; i < n; i += stride) {
        arr[i] = arr[i] * 2.0f;
    }
}

/**
 * Show how Grid-Stride Loop works
 */
__global__ void showStridePattern(int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Only let first few threads print to avoid too much output
    if (idx < 4) {
        printf("Thread %d processes elements: ", idx);
        for (int i = idx; i < n && i < idx + stride * 3; i += stride) {
            printf("%d ", i);
        }
        printf("...\n");
    }
}

int main() {
    printf("========================================\n");
    printf("    Grid-Stride Loop Pattern Demo\n");
    printf("========================================\n\n");

    const int n = 100;
    size_t bytes = n * sizeof(float);

    // Allocate unified memory
    float *data;
    cudaMallocManaged(&data, bytes);

    // ========== Demo 1: Show Stride Pattern ==========
    printf("Demo 1: Stride Pattern Visualization\n");
    printf("Config: <<<2, 4>>> (8 threads processing %d elements)\n", n);
    printf("Each thread processes %d elements\n\n", n / 8);

    showStridePattern<<<2, 4>>>(n);
    cudaDeviceSynchronize();

    // ========== Demo 2: Traditional Approach ==========
    printf("\nDemo 2: Traditional Approach\n");
    printf("Needs <<<(n+255)/256, 256>>> = <<<1, 256>>> threads\n");

    // Initialize data
    for (int i = 0; i < n; i++) {
        data[i] = (float)i;
    }

    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    traditionalProcess<<<blocks, threadsPerBlock>>>(data, n);
    cudaDeviceSynchronize();

    printf("Results (first 10 elements): ");
    for (int i = 0; i < 10; i++) {
        printf("%.0f ", data[i]);
    }
    printf("...\n");

    // ========== Demo 3: Grid-Stride Loop ==========
    printf("\nDemo 3: Grid-Stride Loop\n");
    printf("Using fixed <<<2, 256>>> threads to process %d elements\n", n);

    // Reinitialize data
    for (int i = 0; i < n; i++) {
        data[i] = (float)i;
    }

    gridStrideProcess<<<2, 256>>>(data, n);
    cudaDeviceSynchronize();

    printf("Results (first 10 elements): ");
    for (int i = 0; i < 10; i++) {
        printf("%.0f ", data[i]);
    }
    printf("...\n");

    // ========== Comparison ==========
    printf("\n========================================\n");
    printf("Comparison:\n");
    printf("========================================\n\n");

    printf("Traditional approach:\n");
    printf("  - Need to adjust block count based on n\n");
    printf("  - Block count may exceed limit for large n\n\n");

    printf("Grid-Stride Loop:\n");
    printf("  - Fixed thread configuration\n");
    printf("  - Works for any data size\n");
    printf("  - Better reusability\n");
    printf("  - Better memory access patterns\n");

    // Free memory
    cudaFree(data);

    printf("\n========================================\n");
    printf(" Grid-Stride Loop demo complete!\n");
    printf("========================================\n");

    return 0;
}
