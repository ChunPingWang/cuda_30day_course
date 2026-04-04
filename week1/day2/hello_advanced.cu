#include <stdio.h>

/**
 * Advanced Hello World
 * This version demonstrates multiple threads running simultaneously
 */
__global__ void helloFromGPU() {
    printf("Hello from GPU Thread!\n");
}

/**
 * Version using multiple threads
 */
__global__ void helloFromMultipleThreads() {
    printf("Thread says: Hello World!\n");
}

int main() {
    printf("========================================\n");
    printf("    Advanced CUDA Hello World\n");
    printf("========================================\n\n");

    // Experiment 1: Single thread
    printf("Experiment 1: Launch 1 thread\n");
    printf("----------------------------------------\n");
    helloFromGPU<<<1, 1>>>();
    cudaDeviceSynchronize();

    printf("\nExperiment 2: Launch 5 threads\n");
    printf("----------------------------------------\n");
    helloFromMultipleThreads<<<1, 5>>>();
    cudaDeviceSynchronize();

    printf("\nExperiment 3: Launch 3 Blocks, each Block has 4 threads\n");
    printf("(Total 3 x 4 = 12 threads)\n");
    printf("----------------------------------------\n");
    helloFromMultipleThreads<<<3, 4>>>();
    cudaDeviceSynchronize();

    printf("\n========================================\n");
    printf("Observations:\n");
    printf("- Experiment 1 outputs 1 time\n");
    printf("- Experiment 2 outputs 5 times\n");
    printf("- Experiment 3 outputs 12 times\n");
    printf("\nNote: GPU thread execution order is non-deterministic!\n");
    printf("      You may see different output order each run.\n");
    printf("========================================\n");

    return 0;
}
