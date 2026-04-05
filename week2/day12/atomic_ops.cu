#include <stdio.h>
#include <stdlib.h>

/**
 * 錯誤示範：沒有使用原子操作的計數器
 * ⚠️ 注意：這個函式有 Race Condition（競爭條件），結果會不正確
 */
__global__ void badCounter(int *counter, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // ⚠️ 注意：這兩行不是原子操作！多個執行緒可能同時讀到相同的 temp 值
        // 例如：thread A 和 thread B 都讀到 counter=5，都寫入 6，結果只加了 1 而非 2
        int temp = *counter;   // 讀取
        *counter = temp + 1;   // 寫入（中間可能被其他執行緒插入）
    }
}

/**
 * 正確示範：使用原子操作的計數器
 * atomicAdd 保證「讀取 → 修改 → 寫入」三步驟不會被其他執行緒打斷
 */
__global__ void goodCounter(int *counter, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // atomicAdd(位址, 要加的值)：原子地將 counter 的值加 1
        atomicAdd(counter, 1);
    }
}

/**
 * 直方圖計算：統計每個 bin 出現的次數
 * 多個執行緒可能同時對同一個 bin 加 1，所以必須用原子操作
 */
__global__ void histogram(int *data, int *hist, int n, int numBins) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        int bin = data[idx] % numBins;  // 決定這筆資料屬於哪個 bin
        // &hist[bin] 取 hist[bin] 的位址，原子地加 1
        atomicAdd(&hist[bin], 1);
        // 💡 Debug 提示：如果直方圖總和不等於 n，可能是邊界檢查有誤
    }
}

/**
 * 找最大值：使用 atomicMax 原子地更新最大值
 */
__global__ void findMax(int *data, int *maxVal, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        // atomicMax：如果 data[idx] > *maxVal，就更新 *maxVal（原子操作）
        atomicMax(maxVal, data[idx]);
    }
}

/**
 * 找最小值：使用 atomicMin 原子地更新最小值
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
    // cudaMallocManaged 分配統一記憶體（Unified Memory），CPU 和 GPU 都能直接存取
    // 不需要手動 cudaMemcpy，系統會自動搬移資料（方便但不一定最快）
    cudaMallocManaged(&d_counter, sizeof(int));

    // 錯誤版本
    *d_counter = 0;
    badCounter<<<blocks, threadsPerBlock>>>(d_counter, n);
    // cudaDeviceSynchronize() 等待 GPU 完成，Managed Memory 在同步後才保證 CPU 能讀到最新值
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

    *d_max = 0;          // 最大值初始化為最小可能值
    *d_min = INT_MAX;    // 最小值初始化為最大可能值（INT_MAX = 2147483647）

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

    // 清理：cudaFree 釋放 cudaMalloc / cudaMallocManaged 分配的記憶體
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
