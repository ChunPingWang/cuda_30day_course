#include <stdio.h>
#include <stdlib.h>

/**
 * 錯誤示範：沒有使用原子操作的計數器
 */
__global__ void badCounter(int *counter, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 競爭條件！多個執行緒同時讀寫
        int temp = *counter;
        *counter = temp + 1;
    }
}

/**
 * 正確示範：使用原子操作的計數器
 */
__global__ void goodCounter(int *counter, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicAdd(counter, 1);
    }
}

/**
 * 直方圖計算
 */
__global__ void histogram(int *data, int *hist, int n, int numBins) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        int bin = data[idx] % numBins;
        atomicAdd(&hist[bin], 1);
    }
}

/**
 * 找最大值
 */
__global__ void findMax(int *data, int *maxVal, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        atomicMax(maxVal, data[idx]);
    }
}

/**
 * 找最小值
 */
__global__ void findMin(int *data, int *minVal, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        atomicMin(minVal, data[idx]);
    }
}

int main() {
    printf("========================================\n");
    printf("    原子操作（Atomic Operations）示範\n");
    printf("========================================\n\n");

    const int n = 100000;
    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    // ========== 測試 1：計數器 ==========
    printf("測試 1：計數器（目標值 = %d）\n", n);
    printf("----------------------------------------\n");

    int *d_counter;
    cudaMallocManaged(&d_counter, sizeof(int));

    // 錯誤版本
    *d_counter = 0;
    badCounter<<<blocks, threadsPerBlock>>>(d_counter, n);
    cudaDeviceSynchronize();
    printf("無原子操作: %d（不正確，存在競爭條件）\n", *d_counter);

    // 正確版本
    *d_counter = 0;
    goodCounter<<<blocks, threadsPerBlock>>>(d_counter, n);
    cudaDeviceSynchronize();
    printf("使用 atomicAdd: %d %s\n\n", *d_counter,
           *d_counter == n ? "" : "");

    // ========== 測試 2：直方圖 ==========
    printf("測試 2：直方圖\n");
    printf("----------------------------------------\n");

    const int numBins = 10;
    int *d_data, *d_hist;

    cudaMallocManaged(&d_data, n * sizeof(int));
    cudaMallocManaged(&d_hist, numBins * sizeof(int));

    // 初始化資料
    srand(42);
    for (int i = 0; i < n; i++) {
        d_data[i] = rand() % numBins;
    }

    // 清空直方圖
    for (int i = 0; i < numBins; i++) {
        d_hist[i] = 0;
    }

    histogram<<<blocks, threadsPerBlock>>>(d_data, d_hist, n, numBins);
    cudaDeviceSynchronize();

    printf("直方圖結果:\n");
    int total = 0;
    for (int i = 0; i < numBins; i++) {
        printf("  Bin %d: %d\n", i, d_hist[i]);
        total += d_hist[i];
    }
    printf("  總計: %d（應該 = %d）%s\n\n", total, n,
           total == n ? "" : "");

    // ========== 測試 3：最大最小值 ==========
    printf("測試 3：找最大和最小值\n");
    printf("----------------------------------------\n");

    // 重新生成資料
    for (int i = 0; i < n; i++) {
        d_data[i] = rand() % 10000;
    }

    int *d_max, *d_min;
    cudaMallocManaged(&d_max, sizeof(int));
    cudaMallocManaged(&d_min, sizeof(int));

    *d_max = 0;
    *d_min = INT_MAX;

    findMax<<<blocks, threadsPerBlock>>>(d_data, d_max, n);
    findMin<<<blocks, threadsPerBlock>>>(d_data, d_min, n);
    cudaDeviceSynchronize();

    // CPU 驗證
    int cpuMax = 0, cpuMin = INT_MAX;
    for (int i = 0; i < n; i++) {
        if (d_data[i] > cpuMax) cpuMax = d_data[i];
        if (d_data[i] < cpuMin) cpuMin = d_data[i];
    }

    printf("GPU 最大值: %d (CPU: %d) %s\n", *d_max, cpuMax,
           *d_max == cpuMax ? "" : "");
    printf("GPU 最小值: %d (CPU: %d) %s\n\n", *d_min, cpuMin,
           *d_min == cpuMin ? "" : "");

    // 清理
    cudaFree(d_counter);
    cudaFree(d_data);
    cudaFree(d_hist);
    cudaFree(d_max);
    cudaFree(d_min);

    printf("========================================\n");
    printf("💡 原子操作注意事項：\n");
    printf("1. 原子操作比普通操作慢\n");
    printf("2. 高競爭情況會影響效能\n");
    printf("3. 必要時才使用，避免過度使用\n");
    printf("========================================\n");

    return 0;
}
