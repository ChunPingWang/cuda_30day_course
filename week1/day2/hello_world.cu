#include <stdio.h>

/**
 * 這是我們的第一個 CUDA 核心函數！
 * __global__ 表示這個函數會在 GPU 上執行
 */
__global__ void helloFromGPU() {
    printf("Hello World from GPU!\n");
}

int main() {
    // 步驟 1: 從 CPU 打招呼
    printf("========================================\n");
    printf("        My First CUDA Program\n");
    printf("========================================\n\n");
    printf("Hello World from CPU!\n\n");

    // 步驟 2: 啟動核心函數
    // <<<1, 1>>> 表示：1 個 Block，每個 Block 有 1 個 Thread
    printf("啟動 GPU 核心函數...\n");
    helloFromGPU<<<1, 1>>>();

    // 步驟 3: 等待 GPU 完成工作
    // 因為 CPU 和 GPU 是異步執行的
    cudaDeviceSynchronize();

    printf("\n========================================\n");
    printf("程式執行完成！\n");
    printf("========================================\n");

    return 0;
}
