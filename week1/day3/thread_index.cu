#include <stdio.h>

/**
 * Demonstrate thread index calculation
 */
__global__ void printThreadInfo() {
    // Calculate global index
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread prints its own info
    printf("Block: %2d | Thread: %2d | Global Index: %2d\n",
           blockIdx.x, threadIdx.x, globalIdx);
}

/**
 * Demonstrate 2D indexing
 */
__global__ void print2DThreadInfo() {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    printf("Block(%d,%d) Thread(%d,%d) => Global(%d,%d)\n",
           blockIdx.x, blockIdx.y,
           threadIdx.x, threadIdx.y,
           x, y);
}

int main() {
    printf("========================================\n");
    printf("      CUDA Thread Index Demo\n");
    printf("========================================\n\n");

    // Demo 1: 1D indexing
    printf("Demo 1: 1D Indexing\n");
    printf("Config: <<<3, 4>>> (3 Blocks, 4 Threads each)\n");
    printf("----------------------------------------\n");
    printThreadInfo<<<3, 4>>>();
    cudaDeviceSynchronize();

    printf("\nDemo 2: Different configuration\n");
    printf("Config: <<<2, 8>>> (2 Blocks, 8 Threads each)\n");
    printf("----------------------------------------\n");
    printThreadInfo<<<2, 8>>>();
    cudaDeviceSynchronize();

    // Demo 3: 2D indexing
    printf("\nDemo 3: 2D Indexing\n");
    printf("Config: <<<(2,2), (2,2)>>> (2x2 Block Grid, 2x2 Thread Block)\n");
    printf("----------------------------------------\n");
    dim3 blocks(2, 2);
    dim3 threads(2, 2);
    print2DThreadInfo<<<blocks, threads>>>();
    cudaDeviceSynchronize();

    printf("\n========================================\n");
    printf("Key Observations:\n");
    printf("1. Global Index = blockIdx.x * blockDim.x + threadIdx.x\n");
    printf("2. Thread execution order is non-deterministic!\n");
    printf("3. Different configs can create same number of threads\n");
    printf("========================================\n");

    return 0;
}
