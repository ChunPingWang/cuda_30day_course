#include <stdio.h>
#include <stdlib.h>

/**
 * Day 17: 掃描（Prefix Sum）演算法
 *
 * 展示 Inclusive 和 Exclusive Scan 的實作
 */

#define BLOCK_SIZE 256

// CPU 版本（驗證用）
void cpuInclusiveScan(int *input, int *output, int n) {
    output[0] = input[0];
    for (int i = 1; i < n; i++) {
        output[i] = output[i - 1] + input[i];
    }
}

void cpuExclusiveScan(int *input, int *output, int n) {
    output[0] = 0;
    for (int i = 1; i < n; i++) {
        output[i] = output[i - 1] + input[i - 1];
    }
}

// Warp 級 Inclusive Scan（使用 shuffle）
__device__ int warpInclusiveScan(int val) {
    int lane = threadIdx.x % 32;

    // 使用 warp shuffle 進行掃描
    for (int offset = 1; offset < 32; offset *= 2) {
        int n = __shfl_up_sync(0xffffffff, val, offset);
        if (lane >= offset) {
            val += n;
        }
    }
    return val;
}

// 單一 Block 的 Inclusive Scan
__global__ void blockInclusiveScan(int *input, int *output, int n) {
    __shared__ int sdata[BLOCK_SIZE];
    __shared__ int warpSums[32];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // 載入資料
    int val = (gid < n) ? input[gid] : 0;

    // Warp 級掃描
    int warpResult = warpInclusiveScan(val);

    int lane = tid % 32;
    int warpId = tid / 32;

    // 儲存每個 Warp 的最後一個值
    if (lane == 31) {
        warpSums[warpId] = warpResult;
    }
    __syncthreads();

    // 第一個 Warp 掃描所有 Warp 的總和
    if (warpId == 0) {
        int warpSum = (lane < BLOCK_SIZE / 32) ? warpSums[lane] : 0;
        warpSum = warpInclusiveScan(warpSum);
        warpSums[lane] = warpSum;
    }
    __syncthreads();

    // 加上前面 Warp 的總和
    if (warpId > 0) {
        warpResult += warpSums[warpId - 1];
    }

    // 寫入結果
    if (gid < n) {
        output[gid] = warpResult;
    }
}

// Exclusive Scan = Inclusive Scan 右移一位
__global__ void inclusiveToExclusive(int *inclusive, int *exclusive, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid < n) {
        exclusive[gid] = (gid == 0) ? 0 : inclusive[gid - 1];
    }
}

// Naive 平行掃描（展示用，效率較低）
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

    // GPU 記憶體
    int *d_input, *d_output, *d_exclusive;
    cudaMalloc(&d_input, n * sizeof(int));
    cudaMalloc(&d_output, n * sizeof(int));
    cudaMalloc(&d_exclusive, n * sizeof(int));
    cudaMemcpy(d_input, h_input, n * sizeof(int), cudaMemcpyHostToDevice);

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
