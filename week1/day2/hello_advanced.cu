#include <stdio.h>

/**
 * 進階版 Hello World
 * 這個版本會展示多個執行緒同時運行
 */
__global__ void helloFromGPU() {
    printf("Hello from GPU Thread!\n");
}

/**
 * 使用多個執行緒的版本
 */
__global__ void helloFromMultipleThreads() {
    printf("Thread says: Hello World!\n");
}

int main() {
    printf("========================================\n");
    printf("    進階版 CUDA Hello World\n");
    printf("========================================\n\n");

    // 實驗 1: 單一執行緒
    printf("實驗 1: 啟動 1 個執行緒\n");
    printf("----------------------------------------\n");
    helloFromGPU<<<1, 1>>>();
    cudaDeviceSynchronize();

    printf("\n實驗 2: 啟動 5 個執行緒\n");
    printf("----------------------------------------\n");
    helloFromMultipleThreads<<<1, 5>>>();
    cudaDeviceSynchronize();

    printf("\n實驗 3: 啟動 3 個 Block，每個 Block 有 4 個執行緒\n");
    printf("（總共 3 × 4 = 12 個執行緒）\n");
    printf("----------------------------------------\n");
    helloFromMultipleThreads<<<3, 4>>>();
    cudaDeviceSynchronize();

    printf("\n========================================\n");
    printf("🎯 觀察：\n");
    printf("- 實驗 1 輸出 1 次\n");
    printf("- 實驗 2 輸出 5 次\n");
    printf("- 實驗 3 輸出 12 次\n");
    printf("\n注意：GPU 執行緒的執行順序是不確定的！\n");
    printf("      每次運行可能看到不同的輸出順序。\n");
    printf("========================================\n");

    return 0;
}
