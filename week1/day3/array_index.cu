#include <stdio.h>

/**
 * Use thread index to access array elements
 * Each thread processes one array element
 */
__global__ void fillArray(int *arr, int n) {
    // Calculate global index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure we don't exceed array bounds
    if (idx < n) {
        // Each thread stores its index value * 10 into the array
        arr[idx] = idx * 10;
    }
}

/**
 * Demonstrate the importance of boundary checking
 */
__global__ void printIndexInfo(int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        printf(" Thread %d: Processing array[%d]\n", idx, idx);
    } else {
        printf(" Thread %d: Out of range, idle\n", idx);
    }
}

int main() {
    printf("========================================\n");
    printf("   Using Thread Index to Process Arrays\n");
    printf("========================================\n\n");

    const int n = 10;  // Array size
    int *d_arr;        // Device array pointer

    // Allocate memory on GPU
    cudaMalloc(&d_arr, n * sizeof(int));

    // Kernel configuration
    int threadsPerBlock = 4;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    printf("Array size: %d\n", n);
    printf("Threads per Block: %d\n", threadsPerBlock);
    printf("Number of Blocks: %d\n", blocks);
    printf("Total threads: %d\n\n", blocks * threadsPerBlock);

    // Demo 1: Show which threads will be used
    printf("Demo 1: Thread usage status\n");
    printf("----------------------------------------\n");
    printIndexInfo<<<blocks, threadsPerBlock>>>(n);
    cudaDeviceSynchronize();

    // Demo 2: Fill array
    printf("\nDemo 2: Fill array using GPU\n");
    printf("----------------------------------------\n");
    fillArray<<<blocks, threadsPerBlock>>>(d_arr, n);
    cudaDeviceSynchronize();

    // Copy results back to host and display
    int h_arr[10];
    cudaMemcpy(h_arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Array contents: [ ");
    for (int i = 0; i < n; i++) {
        printf("%d ", h_arr[i]);
    }
    printf("]\n");

    // Free memory
    cudaFree(d_arr);

    printf("\n========================================\n");
    printf("Key Concepts:\n");
    printf("1. idx = blockIdx.x * blockDim.x + threadIdx.x\n");
    printf("2. Must check idx < n to avoid out-of-bounds access\n");
    printf("3. Thread count may exceed data count\n");
    printf("\nExercise:\n");
    printf("Try modifying threadsPerBlock value (e.g., 8, 16)\n");
    printf("Observe how many Blocks and idle threads are needed\n");
    printf("========================================\n");

    return 0;
}
