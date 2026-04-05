#include <stdio.h>
#include <stdlib.h>

/**
 * Day 17: 掃描（Prefix Sum）演算法
 *
 * 展示 Inclusive 和 Exclusive Scan 的實作
 */

#define BLOCK_SIZE 256

// CPU 版本的 Inclusive Scan（驗證 GPU 結果用）
// Inclusive Scan: output[i] = input[0] + input[1] + ... + input[i]（包含自己）
void cpuInclusiveScan(int *input, int *output, int n) {
    output[0] = input[0];
    for (int i = 1; i < n; i++) {
        output[i] = output[i - 1] + input[i];
    }
}

// CPU 版本的 Exclusive Scan
// Exclusive Scan: output[i] = input[0] + input[1] + ... + input[i-1]（不包含自己）
void cpuExclusiveScan(int *input, int *output, int n) {
    output[0] = 0;  // Exclusive Scan 的第一個元素一定是 0（單位元素）
    for (int i = 1; i < n; i++) {
        output[i] = output[i - 1] + input[i - 1];
    }
}

// Warp 級 Inclusive Scan（使用 shuffle 指令）
// __device__ 表示此函式只能在 GPU 上被其他 GPU 函式呼叫（不能從 CPU 呼叫）
__device__ int warpInclusiveScan(int val) {
    int lane = threadIdx.x % 32;  // Warp 內的 lane 編號（0~31），一個 Warp = 32 個執行緒

    // __shfl_up_sync：從同 Warp 中前 offset 個 lane 取值（非常快，不需要共享記憶體）
    // 0xffffffff 表示所有 32 個 lane 都參與
    for (int offset = 1; offset < 32; offset *= 2) {
        int n = __shfl_up_sync(0xffffffff, val, offset);
        if (lane >= offset) {
            val += n;  // 累加前面 lane 的值
        }
    }
    return val;
}

// 單一 Block 的 Inclusive Scan（分三步驟：Warp 內掃描 → Warp 間掃描 → 合併）
__global__ void blockInclusiveScan(int *input, int *output, int n) {
    __shared__ int warpSums[32];  // 共享記憶體：儲存每個 Warp 的總和

    int tid = threadIdx.x;  // 區塊內的執行緒索引
    int gid = blockIdx.x * blockDim.x + tid;  // 全域索引

    // 步驟 0：載入資料（超出範圍的補 0）
    int val = (gid < n) ? input[gid] : 0;

    // 步驟 1：每個 Warp 內部做 Inclusive Scan
    int warpResult = warpInclusiveScan(val);

    int lane = tid % 32;    // Warp 內的 lane 編號
    int warpId = tid / 32;  // 第幾個 Warp

    // 每個 Warp 的最後一個 lane（lane 31）持有該 Warp 的總和
    if (lane == 31) {
        warpSums[warpId] = warpResult;
    }
    __syncthreads();  // ⚠️ 注意：跨 Warp 溝通必須用 __syncthreads 同步

    // 步驟 2：用第一個 Warp 對所有 Warp 的總和做掃描
    if (warpId == 0) {
        int warpSum = (lane < BLOCK_SIZE / 32) ? warpSums[lane] : 0;
        warpSum = warpInclusiveScan(warpSum);
        warpSums[lane] = warpSum;
    }
    __syncthreads();

    // 步驟 3：每個元素加上「前面所有 Warp 的總和」
    if (warpId > 0) {
        warpResult += warpSums[warpId - 1];
    }

    // 寫入結果
    if (gid < n) {
        output[gid] = warpResult;
    }
}

// 將 Inclusive Scan 轉換成 Exclusive Scan（整個陣列往右移一格，第一個填 0）
// 例如 Inclusive=[1,3,6,10] → Exclusive=[0,1,3,6]
__global__ void inclusiveToExclusive(int *inclusive, int *exclusive, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid < n) {
        exclusive[gid] = (gid == 0) ? 0 : inclusive[gid - 1];
    }
}

// Naive 平行掃描（展示用，效率較低）
// ⚠️ 注意：這個方法有 race condition 風險（讀寫同一陣列），需要多次呼叫且用雙緩衝才正確
__global__ void naiveInclusiveScan(int *data, int n, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n && idx >= stride) {
        data[idx] += data[idx - stride];
    }
}

void printArray(const char *name, int *arr, int n) {
    printf("%s: [", name);
    for (int i = 0; i < n && i < 16; i++) {
        printf("%d", arr[i]);
        if (i < n - 1 && i < 15) printf(", ");
    }
    if (n > 16) printf(", ...");
    printf("]\n");
}

int main() {
    printf("========================================\n");
    printf("    掃描（Prefix Sum）演算法\n");
    printf("========================================\n\n");

    const int n = 16;  // 小陣列便於觀察
    int *h_input = (int*)malloc(n * sizeof(int));
    int *h_output = (int*)malloc(n * sizeof(int));
    int *h_expected = (int*)malloc(n * sizeof(int));

    // 初始化輸入
    printf("初始化輸入資料...\n");
    for (int i = 0; i < n; i++) {
        h_input[i] = i + 1;  // [1, 2, 3, ..., n]
    }
    printArray("輸入", h_input, n);
    printf("\n");

    // 分配 GPU 記憶體
    int *d_input, *d_output, *d_exclusive;
    cudaMalloc(&d_input, n * sizeof(int));      // GPU 上的輸入陣列
    cudaMalloc(&d_output, n * sizeof(int));     // GPU 上的 Inclusive Scan 結果
    cudaMalloc(&d_exclusive, n * sizeof(int));  // GPU 上的 Exclusive Scan 結果
    cudaMemcpy(d_input, h_input, n * sizeof(int), cudaMemcpyHostToDevice);  // CPU → GPU

    // ========== Inclusive Scan ==========
    printf("【Inclusive Scan】\n");
    printf("定義: output[i] = input[0] + input[1] + ... + input[i]\n\n");

    // CPU 版本
    cpuInclusiveScan(h_input, h_expected, n);
    printArray("CPU 結果", h_expected, n);

    // GPU 版本
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    blockInclusiveScan<<<blocks, BLOCK_SIZE>>>(d_input, d_output, n);
    cudaMemcpy(h_output, d_output, n * sizeof(int), cudaMemcpyDeviceToHost);
    printArray("GPU 結果", h_output, n);

    // 驗證
    bool correct = true;
    for (int i = 0; i < n; i++) {
        if (h_output[i] != h_expected[i]) {
            correct = false;
            break;
        }
    }
    printf("驗證: %s\n\n", correct ? "通過" : "失敗");

    // ========== Exclusive Scan ==========
    printf("【Exclusive Scan】\n");
    printf("定義: output[i] = input[0] + input[1] + ... + input[i-1]\n");
    printf("     output[0] = 0 (identity element)\n\n");

    // CPU 版本
    cpuExclusiveScan(h_input, h_expected, n);
    printArray("CPU 結果", h_expected, n);

    // GPU 版本（從 Inclusive 轉換）
    inclusiveToExclusive<<<blocks, BLOCK_SIZE>>>(d_output, d_exclusive, n);
    cudaMemcpy(h_output, d_exclusive, n * sizeof(int), cudaMemcpyDeviceToHost);
    printArray("GPU 結果", h_output, n);

    // 驗證
    correct = true;
    for (int i = 0; i < n; i++) {
        if (h_output[i] != h_expected[i]) {
            correct = false;
            break;
        }
    }
    printf("驗證: %s\n\n", correct ? "通過" : "失敗");

    // ========== 應用範例 ==========
    printf("【應用：計算每個元素前面有多少個元素】\n");
    printf("這就是 Exclusive Scan 的結果！\n");
    printf("例如: 第 5 個元素（index=4）前面有 %d 個元素\n\n", h_output[4]);

    printf("【應用：累積和】\n");
    printf("Inclusive Scan 直接給出累積和。\n");
    cudaMemcpy(h_output, d_output, n * sizeof(int), cudaMemcpyDeviceToHost);
    printf("總和 = output[n-1] = %d\n", h_output[n-1]);
    printf("驗證: 1+2+...+%d = %d*%d/2 = %d\n\n", n, n, n+1, n*(n+1)/2);

    // 清理
    free(h_input);
    free(h_output);
    free(h_expected);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_exclusive);

    printf("========================================\n");
    printf("關鍵概念：\n");
    printf("1. Inclusive: 包含當前元素\n");
    printf("2. Exclusive: 不包含當前元素（從 0 開始）\n");
    printf("3. 平行掃描是許多平行演算法的基礎\n");
    printf("4. Warp shuffle 可以實現高效的 warp 級掃描\n");
    printf("========================================\n");

    return 0;
}
