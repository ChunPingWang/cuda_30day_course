#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

/**
 * CUDA kernel function: Vector Addition
 */
// __global__ 表示此函式在 GPU 上執行，由 CPU 端呼叫
__global__ void vectorAddGPU(float *a, float *b, float *c, int n) {
    // 計算全域執行緒索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // ⚠️ 注意：邊界檢查，避免存取超出陣列範圍的記憶體
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

/**
 * CPU 版本：向量加法（用來和 GPU 版本做效能比較）
 */
void vectorAddCPU(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

/**
 * Initialize array
 */
void initArray(float *arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = (float)(rand() % 100) / 10.0f;
    }
}

/**
 * 驗證 GPU 和 CPU 的計算結果是否一致
 */
bool verifyResult(float *gpu, float *cpu, int n) {
    for (int i = 0; i < n; i++) {
        // 💡 Debug 提示：浮點數比較要用容差（epsilon），不能直接用 ==
        if (fabs(gpu[i] - cpu[i]) > 0.001f) {
            return false;
        }
    }
    return true;
}

int main() {
    printf("========================================\n");
    printf("    CPU vs GPU Performance Comparison\n");
    printf("========================================\n\n");

    // Test different array sizes
    int sizes[] = {1000, 10000, 100000, 1000000, 10000000};
    int numSizes = 5;

    srand((unsigned int)time(NULL));

    printf("%-15s %-15s %-15s %-10s\n",
           "Array Size", "CPU Time(ms)", "GPU Time(ms)", "Speedup");
    printf("--------------------------------------------------------\n");

    for (int s = 0; s < numSizes; s++) {
        int n = sizes[s];
        size_t bytes = n * sizeof(float);

        // Allocate host memory
        float *h_A = (float*)malloc(bytes);
        float *h_B = (float*)malloc(bytes);
        float *h_C_cpu = (float*)malloc(bytes);
        float *h_C_gpu = (float*)malloc(bytes);

        // Initialize data
        initArray(h_A, n);
        initArray(h_B, n);

        // ========== CPU Timing ==========
        clock_t cpuStart = clock();
        vectorAddCPU(h_A, h_B, h_C_cpu, n);
        clock_t cpuEnd = clock();
        double cpuTime = ((double)(cpuEnd - cpuStart)) / CLOCKS_PER_SEC * 1000;

        // ========== GPU 計時 ==========
        float *d_A, *d_B, *d_C;
        // 在 GPU 上配置記憶體
        cudaMalloc(&d_A, bytes);
        cudaMalloc(&d_B, bytes);
        cudaMalloc(&d_C, bytes);

        // 建立 CUDA 事件來精確計時（比 clock() 更準確）
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // 將資料從 CPU 複製到 GPU
        cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

        // 設定 kernel 啟動參數
        int threadsPerBlock = 256;  // 每個 block 256 個執行緒（常見的設定值）
        int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;  // 無條件進位除法

        // 開始計時
        cudaEventRecord(start);

        // 用 <<<blocks, threadsPerBlock>>> 語法啟動 kernel
        vectorAddGPU<<<blocks, threadsPerBlock>>>(d_A, d_B, d_C, n);

        // 停止計時並等待 GPU 完成
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);  // 等待 stop 事件完成

        float gpuTime;
        cudaEventElapsedTime(&gpuTime, start, stop);  // 計算兩個事件之間的毫秒數

        // Copy results back to CPU
        cudaMemcpy(h_C_gpu, d_C, bytes, cudaMemcpyDeviceToHost);

        // Verify results
        bool correct = verifyResult(h_C_gpu, h_C_cpu, n);

        // 計算加速比：CPU 時間 / GPU 時間
        // 💡 Debug 提示：如果 speedup < 1，代表 GPU 反而比較慢（小陣列常見，因為有資料傳輸成本）
        double speedup = cpuTime / gpuTime;

        // Output results
        printf("%-15d %-15.3f %-15.3f %-10.2fx %s\n",
               n, cpuTime, gpuTime, speedup,
               correct ? "[OK]" : "[FAIL]");

        // Cleanup
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        free(h_A);
        free(h_B);
        free(h_C_cpu);
        free(h_C_gpu);
    }

    printf("\n========================================\n");
    printf("Observations:\n");
    printf("1. GPU may not be faster for small arrays\n");
    printf("   (due to data transfer overhead)\n");
    printf("2. GPU advantage is significant for large arrays\n");
    printf("3. Speedup increases with array size\n");
    printf("========================================\n");

    return 0;
}
