#include <stdio.h>
#include <time.h>

#define N 10000000
#define THREADS_PER_BLOCK 256

/**
 * 有 Warp Divergence 的版本
 * 每個執行緒根據奇偶執行不同的操作
 * ⚠️ 注意：同一個 Warp 中的執行緒走不同 if/else 分支時，會造成 Warp Divergence
 *    GPU 必須序列化執行兩條路徑，效能會下降
 */
// __global__ 表示這是 GPU 核心函式
__global__ void withDivergence(float *data, int n) {
    // 計算全域執行緒索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {  // 邊界檢查，避免存取超出陣列範圍
        // ⚠️ 注意：相鄰執行緒（如 thread 0 和 thread 1）走不同分支 → Warp Divergence
        if (idx % 2 == 0) {
            // 偶數索引：計算 sin
            for (int i = 0; i < 10; i++) {
                data[idx] = sinf(data[idx]);
            }
        } else {
            // 奇數索引：計算 cos
            for (int i = 0; i < 10; i++) {
                data[idx] = cosf(data[idx]);
            }
        }
    }
}

/**
 * 減少 Warp Divergence 的版本
 * 策略：先讓所有執行緒處理偶數，再處理奇數，同一 Warp 內執行相同路徑
 */
__global__ void reducedDivergence(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int halfN = n / 2;

    if (idx < halfN) {
        // 第一階段：所有執行緒一起處理偶數索引（同一條路徑，無 divergence）
        int evenIdx = idx * 2;
        for (int i = 0; i < 10; i++) {
            data[evenIdx] = sinf(data[evenIdx]);
        }
    }

    // __syncthreads() 同步 Block 內所有執行緒，確保第一階段完成
    // ⚠️ 注意：__syncthreads() 只能同步同一個 Block 內的執行緒，不能跨 Block
    __syncthreads();

    if (idx < halfN) {
        // 第二階段：所有執行緒一起處理奇數索引
        int oddIdx = idx * 2 + 1;
        for (int i = 0; i < 10; i++) {
            data[oddIdx] = cosf(data[oddIdx]);
        }
    }
}

/**
 * 無 Divergence 的版本（所有執行緒做相同的事）
 */
__global__ void noDivergence(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        // 所有執行緒執行相同的操作
        for (int i = 0; i < 10; i++) {
            data[idx] = sinf(data[idx]);
        }
    }
}

int main() {
    printf("========================================\n");
    printf("    Warp Divergence 效能比較\n");
    printf("========================================\n\n");

    size_t bytes = N * sizeof(float);

    // 在 GPU 上分配記憶體（d_ 前綴代表 device，即 GPU 端）
    float *d_data;
    cudaMalloc(&d_data, bytes);  // 在 GPU 全域記憶體中分配 bytes 大小的空間

    // 計算需要多少個 Block（向上取整，確保涵蓋所有元素）
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms;

    // ========== 測試 1: 有 Divergence ==========
    printf("測試 1: 有 Warp Divergence\n");
    printf("  if (idx %% 2 == 0) sin else cos\n");

    // 初始化
    cudaMemset(d_data, 0, bytes);

    cudaEventRecord(start);
    withDivergence<<<blocks, THREADS_PER_BLOCK>>>(d_data, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("  時間: %.3f ms\n\n", ms);
    float divergentTime = ms;

    // ========== 測試 2: 減少 Divergence ==========
    printf("測試 2: 減少 Warp Divergence\n");
    printf("  先處理所有偶數，再處理所有奇數\n");

    cudaMemset(d_data, 0, bytes);

    cudaEventRecord(start);
    reducedDivergence<<<blocks, THREADS_PER_BLOCK>>>(d_data, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("  時間: %.3f ms\n\n", ms);
    float reducedTime = ms;

    // ========== 測試 3: 無 Divergence ==========
    printf("測試 3: 無 Warp Divergence\n");
    printf("  所有執行緒執行相同操作\n");

    cudaMemset(d_data, 0, bytes);

    cudaEventRecord(start);
    noDivergence<<<blocks, THREADS_PER_BLOCK>>>(d_data, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("  時間: %.3f ms\n\n", ms);
    float noDivTime = ms;

    // ========== 結果分析 ==========
    printf("========================================\n");
    printf("效能比較:\n");
    printf("  有 Divergence:    %.3f ms (基準)\n", divergentTime);
    printf("  減少 Divergence:  %.3f ms (%.1fx)\n",
           reducedTime, divergentTime / reducedTime);
    printf("  無 Divergence:    %.3f ms (%.1fx)\n",
           noDivTime, divergentTime / noDivTime);
    printf("\n");
    printf("💡 觀察:\n");
    printf("  - Warp Divergence 會降低效能\n");
    printf("  - 重新組織計算可以減少分歧\n");
    printf("  - 最好讓同一個 Warp 的執行緒執行相同路徑\n");
    printf("========================================\n");

    // 清理：釋放 GPU 記憶體
    cudaFree(d_data);  // 釋放之前 cudaMalloc 分配的 GPU 記憶體
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
