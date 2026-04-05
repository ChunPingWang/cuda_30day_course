#include <stdio.h>

/**
 * 這是我們的第一個 CUDA kernel 函式！
 * __global__ 關鍵字表示：這個函式由 CPU 呼叫，但在 GPU 上執行
 */
__global__ void helloFromGPU() {
    printf("Hello World from GPU!\n");
}

// main() 在 CPU 上執行，是程式的進入點
int main() {
    // 第一步：CPU 端印出訊息
    printf("========================================\n");
    printf("        My First CUDA Program\n");
    printf("========================================\n\n");
    printf("Hello World from CPU!\n\n");

    // 第二步：啟動 GPU kernel 函式
    // <<<1, 1>>> 是 CUDA 特有語法，意思是：1 個 Block、每個 Block 有 1 個 Thread
    // ⚠️ 注意：<<<>>> 裡的數字決定了有多少個 GPU 執行緒會同時執行這個函式
    printf("Launching GPU kernel function...\n");
    helloFromGPU<<<1, 1>>>();

    // 第三步：等待 GPU 執行完畢
    // ⚠️ 注意：CPU 和 GPU 是「非同步」執行的，如果不等待，CPU 可能在 GPU 還沒印完就結束程式了
    // 💡 Debug 提示：如果看不到 GPU 的輸出，很可能是忘了加這行
    cudaDeviceSynchronize();

    printf("\n========================================\n");
    printf("Program execution complete!\n");
    printf("========================================\n");

    return 0;
}
