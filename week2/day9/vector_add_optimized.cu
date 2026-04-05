#include <stdio.h>
#include <stdlib.h>

#define N 10000000
#define THREADS_PER_BLOCK 256

/**
 * 版本 1：基本版本
 * 每個執行緒負責計算一個元素的加法
 */
__global__ void vectorAddBasic(float *a, float *b, float *c, int n) {
    // 計算此執行緒負責的全域索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // blockIdx.x=Block編號, blockDim.x=Block大小, threadIdx.x=執行緒編號
    if (idx < n) {  // ⚠️ 注意：邊界檢查很重要，最後一個 Block 可能超出陣列範圍
        c[idx] = a[idx] + b[idx];
    }
}

/**
 * 版本 2：使用 float4（每個執行緒處理 4 個元素）
 * float4 是 CUDA 內建的向量類型，一次讀取 4 個 float（128 bits），提高記憶體頻寬利用率
 */
__global__ void vectorAddFloat4(float4 *a, float4 *b, float4 *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n / 4) {
        // 一次載入 4 個 float（合併記憶體存取，減少記憶體交易次數）
        float4 va = a[idx];
        float4 vb = b[idx];

        // float4 有 .x .y .z .w 四個分量
        float4 vc;
        vc.x = va.x + vb.x;
        vc.y = va.y + vb.y;
        vc.z = va.z + vb.z;
        vc.w = va.w + vb.w;

        c[idx] = vc;  // 一次寫入 4 個結果
    }
    // ⚠️ 注意：N 必須是 4 的倍數，否則尾端的元素不會被處理
}

/**
 * 版本 3：Grid-Stride Loop + 迴圈展開（Loop Unrolling）
 * Grid-Stride：每個執行緒用固定步幅跳躍處理多個元素，適合資料量 >> 執行緒數的情境
 */
__global__ void vectorAddStrideUnroll(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // gridDim.x 是 Grid 中 Block 的數量；stride = 所有執行緒的總數
    int stride = blockDim.x * gridDim.x;

    // 每個執行緒處理多個元素，每次處理 4 個（迴圈展開，減少迴圈控制的指令開銷）
    for (int i = idx * 4; i < n; i += stride * 4) {
        if (i + 3 < n) {
            c[i]     = a[i]     + b[i];
            c[i + 1] = a[i + 1] + b[i + 1];
            c[i + 2] = a[i + 2] + b[i + 2];
            c[i + 3] = a[i + 3] + b[i + 3];
        }
    }
}

void initArray(float *arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = (float)(rand() % 100) / 10.0f;
    }
}

bool verify(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        if (abs(c[i] - (a[i] + b[i])) > 0.001f) {
            return false;
        }
    }
    return true;
}

int main() {
    printf("========================================\n");
    printf("    向量加法最佳化比較\n");
    printf("    N = %d\n", N);
    printf("========================================\n\n");

    size_t bytes = N * sizeof(float);

    // 分配主機（CPU）記憶體（h_ 前綴代表 host）
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);

    // 分配裝置（GPU）記憶體（d_ 前綴代表 device）
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);  // 在 GPU 上分配記憶體
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    // 💡 Debug 提示：cudaMalloc 失敗時回傳 cudaErrorMemoryAllocation，建議加錯誤檢查

    // 初始化
    srand(42);
    initArray(h_a, N);
    initArray(h_b, N);

    // 將資料從 CPU 複製到 GPU（cudaMemcpyHostToDevice = CPU → GPU）
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms;

    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // ========== 版本 1：基本版本 ==========
    printf("版本 1：基本版本\n");
    printf("  每個執行緒處理 1 個元素\n");

    cudaEventRecord(start);
    vectorAddBasic<<<blocks, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);

    // 將結果從 GPU 複製回 CPU（cudaMemcpyDeviceToHost = GPU → CPU）
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    bool correct = verify(h_a, h_b, h_c, N);

    printf("  時間: %.3f ms %s\n\n", ms, correct ? "[OK]" : "[FAIL]");
    float baseTime = ms;

    // ========== 版本 2：float4 ==========
    printf("版本 2：使用 float4\n");
    printf("  每個執行緒處理 4 個元素\n");

    int blocks4 = (N / 4 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    cudaEventRecord(start);
    vectorAddFloat4<<<blocks4, THREADS_PER_BLOCK>>>(
        (float4*)d_a, (float4*)d_b, (float4*)d_c, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    correct = verify(h_a, h_b, h_c, N);

    printf("  時間: %.3f ms (%.2fx) %s\n\n", ms, baseTime/ms, correct ? "[OK]" : "[FAIL]");

    // ========== 版本 3：Grid-Stride + 展開 ==========
    printf("版本 3：Grid-Stride + 展開\n");
    printf("  固定 Block 數，每個執行緒處理多個元素\n");

    int fixedBlocks = 256;  // 固定 Block 數

    cudaEventRecord(start);
    vectorAddStrideUnroll<<<fixedBlocks, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    correct = verify(h_a, h_b, h_c, N);

    printf("  時間: %.3f ms (%.2fx) %s\n\n", ms, baseTime/ms, correct ? "[OK]" : "[FAIL]");

    // 清理：釋放 GPU 和 CPU 記憶體
    cudaFree(d_a);   // 釋放 GPU 記憶體
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);       // 釋放 CPU 記憶體
    free(h_b);
    free(h_c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("========================================\n");
    printf("* 最佳化技巧：\n");
    printf("1. 使用向量類型 (float4) 減少記憶體交易\n");
    printf("2. 迴圈展開減少指令開銷\n");
    printf("3. Grid-Stride 模式提高靈活性\n");
    printf("========================================\n");

    return 0;
}
