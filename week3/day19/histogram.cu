#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/**
 * Day 19: 直方圖計算
 *
 * 比較三種不同的直方圖實作方法
 */

#define NUM_BINS 256
#define BLOCK_SIZE 256

// 方法 1：全域記憶體原子操作
__global__ void histogramGlobal(unsigned char *data, unsigned int *hist, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        atomicAdd(&hist[data[idx]], 1);
    }
}

// 方法 2：共享記憶體優化
__global__ void histogramShared(unsigned char *data, unsigned int *hist, int n) {
    __shared__ unsigned int localHist[NUM_BINS];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 初始化共享記憶體（多個執行緒協作）
    for (int i = tid; i < NUM_BINS; i += blockDim.x) {
        localHist[i] = 0;
    }
    __syncthreads();

    // 累積到局部直方圖
    if (idx < n) {
        atomicAdd(&localHist[data[idx]], 1);
    }
    __syncthreads();

    // 合併到全域直方圖
    for (int i = tid; i < NUM_BINS; i += blockDim.x) {
        if (localHist[i] > 0) {
            atomicAdd(&hist[i], localHist[i]);
        }
    }
}

// 方法 3：每個執行緒處理多個元素
#define COARSE_FACTOR 4

__global__ void histogramCoarsened(unsigned char *data, unsigned int *hist, int n) {
    __shared__ unsigned int localHist[NUM_BINS];

    int tid = threadIdx.x;
    int idx = (blockIdx.x * blockDim.x + tid) * COARSE_FACTOR;

    // 初始化
    for (int i = tid; i < NUM_BINS; i += blockDim.x) {
        localHist[i] = 0;
    }
    __syncthreads();

    // 每個執行緒處理 COARSE_FACTOR 個元素
    for (int i = 0; i < COARSE_FACTOR; i++) {
        if (idx + i < n) {
            atomicAdd(&localHist[data[idx + i]], 1);
        }
    }
    __syncthreads();

    // 合併
    for (int i = tid; i < NUM_BINS; i += blockDim.x) {
        if (localHist[i] > 0) {
            atomicAdd(&hist[i], localHist[i]);
        }
    }
}

// CPU 版本（驗證用）
void histogramCPU(unsigned char *data, unsigned int *hist, int n) {
    for (int i = 0; i < NUM_BINS; i++) {
        hist[i] = 0;
    }
    for (int i = 0; i < n; i++) {
        hist[data[i]]++;
    }
}

// 計時輔助函數
float timeKernel(void (*kernel)(unsigned char*, unsigned int*, int),
                 unsigned char *d_data, unsigned int *d_hist, int n,
                 int blocks, int threads) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 清空直方圖
    cudaMemset(d_hist, 0, NUM_BINS * sizeof(unsigned int));

    // 暖機
    kernel<<<blocks, threads>>>(d_data, d_hist, n);
    cudaDeviceSynchronize();

    // 計時
    cudaMemset(d_hist, 0, NUM_BINS * sizeof(unsigned int));
    cudaEventRecord(start);

    for (int i = 0; i < 100; i++) {
        cudaMemset(d_hist, 0, NUM_BINS * sizeof(unsigned int));
        kernel<<<blocks, threads>>>(d_data, d_hist, n);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms / 100.0f;
}

bool verifyHistogram(unsigned int *h_hist, unsigned int *d_hist_result, int n) {
    for (int i = 0; i < NUM_BINS; i++) {
        if (h_hist[i] != d_hist_result[i]) {
            printf("Mismatch at bin %d: CPU=%u, GPU=%u\n",
                   i, h_hist[i], d_hist_result[i]);
            return false;
        }
    }
    return true;
}

int main() {
    printf("========================================\n");
    printf("    直方圖計算 - 三種方法比較\n");
    printf("========================================\n\n");

    const int n = 1 << 24;  // 16M 元素
    printf("資料大小: %d 個元素 (%.1f MB)\n", n, n / (1024.0f * 1024.0f));
    printf("Bins 數量: %d\n\n", NUM_BINS);

    // 分配主機記憶體
    unsigned char *h_data = (unsigned char*)malloc(n);
    unsigned int *h_hist = (unsigned int*)malloc(NUM_BINS * sizeof(unsigned int));
    unsigned int *h_result = (unsigned int*)malloc(NUM_BINS * sizeof(unsigned int));

    // 初始化隨機資料
    srand(42);
    for (int i = 0; i < n; i++) {
        h_data[i] = rand() % NUM_BINS;
    }

    // CPU 計算（驗證用）
    histogramCPU(h_data, h_hist, n);

    // 分配設備記憶體
    unsigned char *d_data;
    unsigned int *d_hist;
    cudaMalloc(&d_data, n);
    cudaMalloc(&d_hist, NUM_BINS * sizeof(unsigned int));
    cudaMemcpy(d_data, h_data, n, cudaMemcpyHostToDevice);

    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    printf("執行效能測試...\n\n");

    // 方法 1：全域原子操作
    float time1 = timeKernel(histogramGlobal, d_data, d_hist, n, blocks, BLOCK_SIZE);
    cudaMemcpy(h_result, d_hist, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    bool correct1 = verifyHistogram(h_hist, h_result, n);

    // 方法 2：共享記憶體
    float time2 = timeKernel(histogramShared, d_data, d_hist, n, blocks, BLOCK_SIZE);
    cudaMemcpy(h_result, d_hist, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    bool correct2 = verifyHistogram(h_hist, h_result, n);

    // 方法 3：粗粒度
    int blocksCoarse = (n / COARSE_FACTOR + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cudaMemset(d_hist, 0, NUM_BINS * sizeof(unsigned int));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    histogramCoarsened<<<blocksCoarse, BLOCK_SIZE>>>(d_data, d_hist, n);
    cudaDeviceSynchronize();

    cudaMemset(d_hist, 0, NUM_BINS * sizeof(unsigned int));
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        cudaMemset(d_hist, 0, NUM_BINS * sizeof(unsigned int));
        histogramCoarsened<<<blocksCoarse, BLOCK_SIZE>>>(d_data, d_hist, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time3 = 0;
    cudaEventElapsedTime(&time3, start, stop);
    time3 /= 100.0f;

    cudaMemcpy(h_result, d_hist, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    bool correct3 = verifyHistogram(h_hist, h_result, n);

    // 結果
    printf("結果比較：\n");
    printf("----------------------------------------\n");
    printf("方法                    時間(ms)   驗證\n");
    printf("----------------------------------------\n");
    printf("1. 全域原子操作         %7.3f    %s\n", time1, correct1 ? "通過" : "失敗");
    printf("2. 共享記憶體           %7.3f    %s\n", time2, correct2 ? "通過" : "失敗");
    printf("3. 粗粒度(x%d)          %7.3f    %s\n", COARSE_FACTOR, time3, correct3 ? "通過" : "失敗");
    printf("----------------------------------------\n\n");

    printf("加速比（相對於全域原子操作）：\n");
    printf("  共享記憶體: %.2fx\n", time1 / time2);
    printf("  粗粒度:     %.2fx\n", time1 / time3);
    printf("\n");

    // 顯示部分直方圖結果
    printf("直方圖前 10 個 bins：\n");
    for (int i = 0; i < 10; i++) {
        printf("  bin[%3d] = %u\n", i, h_hist[i]);
    }

    // 計算統計資訊
    unsigned int total = 0;
    unsigned int maxCount = 0;
    int maxBin = 0;
    for (int i = 0; i < NUM_BINS; i++) {
        total += h_hist[i];
        if (h_hist[i] > maxCount) {
            maxCount = h_hist[i];
            maxBin = i;
        }
    }
    printf("\n統計資訊：\n");
    printf("  總計: %u\n", total);
    printf("  最常見的值: %d (出現 %u 次)\n", maxBin, maxCount);
    printf("  平均每個 bin: %.1f\n", (float)total / NUM_BINS);

    // 清理
    free(h_data);
    free(h_hist);
    free(h_result);
    cudaFree(d_data);
    cudaFree(d_hist);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
