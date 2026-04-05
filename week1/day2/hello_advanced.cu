#include <stdio.h>

/**
 * 進階 Hello World
 * 示範多個執行緒同時在 GPU 上執行
 */
// __global__ 表示這是一個 kernel 函式（由 CPU 呼叫、在 GPU 上執行）
__global__ void helloFromGPU() {
    printf("Hello from GPU Thread!\n");
}

/**
 * 多執行緒版本的 kernel
 */
__global__ void helloFromMultipleThreads() {
    printf("Thread says: Hello World!\n");
}

int main() {
    printf("========================================\n");
    printf("    Advanced CUDA Hello World\n");
    printf("========================================\n\n");

    // 實驗 1：單一執行緒
    printf("Experiment 1: Launch 1 thread\n");
    printf("----------------------------------------\n");
    helloFromGPU<<<1, 1>>>(); // <<<1, 1>>> = 1 個 Block、1 個 Thread
    cudaDeviceSynchronize();  // 等待 GPU 完成，才能看到輸出

    // 實驗 2：1 個 Block 裡面有 5 個 Thread
    printf("\nExperiment 2: Launch 5 threads\n");
    printf("----------------------------------------\n");
    helloFromMultipleThreads<<<1, 5>>>(); // <<<1, 5>>> = 1 個 Block、5 個 Thread
    cudaDeviceSynchronize();

    // 實驗 3：3 個 Block，每個 Block 有 4 個 Thread，共 3x4=12 個執行緒
    printf("\nExperiment 3: Launch 3 Blocks, each Block has 4 threads\n");
    printf("(Total 3 x 4 = 12 threads)\n");
    printf("----------------------------------------\n");
    helloFromMultipleThreads<<<3, 4>>>(); // <<<Block數, 每Block的Thread數>>>
    cudaDeviceSynchronize();
    // ⚠️ 注意：每次執行的輸出順序可能不同，因為 GPU 執行緒是平行且不保證順序的

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
