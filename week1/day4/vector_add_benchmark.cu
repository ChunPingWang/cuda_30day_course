#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/**
 * CUDA 核心函數：向量加法
 */
__global__ void vectorAddGPU(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

/**
 * CPU 版本：向量加法
 */
void vectorAddCPU(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

/**
 * 初始化陣列
 */
void initArray(float *arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = (float)(rand() % 100) / 10.0f;
    }
}

/**
 * 驗證結果
 */
bool verifyResult(float *gpu, float *cpu, int n) {
    for (int i = 0; i < n; i++) {
        if (abs(gpu[i] - cpu[i]) > 0.001f) {
            return false;
        }
    }
    return true;
}

int main() {
    printf("========================================\n");
    printf("    CPU vs GPU 效能比較\n");
    printf("========================================\n\n");

    // 測試不同的陣列大小
    int sizes[] = {1000, 10000, 100000, 1000000, 10000000};
    int numSizes = 5;

    srand(time(NULL));

    printf("%-15s %-15s %-15s %-10s\n",
           "陣列大小", "CPU 時間(ms)", "GPU 時間(ms)", "加速比");
    printf("--------------------------------------------------------\n");

    for (int s = 0; s < numSizes; s++) {
        int n = sizes[s];
        size_t bytes = n * sizeof(float);

        // 分配主機記憶體
        float *h_A = (float*)malloc(bytes);
        float *h_B = (float*)malloc(bytes);
        float *h_C_cpu = (float*)malloc(bytes);
        float *h_C_gpu = (float*)malloc(bytes);

        // 初始化資料
        initArray(h_A, n);
        initArray(h_B, n);

        // ========== CPU 計時 ==========
        clock_t cpuStart = clock();
        vectorAddCPU(h_A, h_B, h_C_cpu, n);
        clock_t cpuEnd = clock();
        double cpuTime = ((double)(cpuEnd - cpuStart)) / CLOCKS_PER_SEC * 1000;

        // ========== GPU 計時 ==========
        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, bytes);
        cudaMalloc(&d_B, bytes);
        cudaMalloc(&d_C, bytes);

        // 創建 CUDA 事件來計時
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // 複製資料到 GPU
        cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

        // 配置核心函數
        int threadsPerBlock = 256;
        int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

        // 計時開始
        cudaEventRecord(start);

        // 執行核心函數
        vectorAddGPU<<<blocks, threadsPerBlock>>>(d_A, d_B, d_C, n);

        // 計時結束
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float gpuTime;
        cudaEventElapsedTime(&gpuTime, start, stop);

        // 複製結果回 CPU
        cudaMemcpy(h_C_gpu, d_C, bytes, cudaMemcpyDeviceToHost);

        // 驗證結果
        bool correct = verifyResult(h_C_gpu, h_C_cpu, n);

        // 計算加速比
        double speedup = cpuTime / gpuTime;

        // 輸出結果
        printf("%-15d %-15.3f %-15.3f %-10.2fx %s\n",
               n, cpuTime, gpuTime, speedup,
               correct ? "✓" : "✗");

        // 清理
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
    printf("💡 觀察：\n");
    printf("1. 小陣列時 GPU 可能不比 CPU 快\n");
    printf("   （因為資料傳輸需要時間）\n");
    printf("2. 大陣列時 GPU 優勢明顯\n");
    printf("3. 加速比隨陣列大小增加\n");
    printf("========================================\n");

    return 0;
}
