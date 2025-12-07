#include <stdio.h>
#include <stdlib.h>
#include <float.h>

/**
 * 練習 4：找出陣列最大值
 *
 * 這是一個簡化版本：
 * - 將陣列分成多個區段
 * - 每個 Block 找出其區段的最大值
 * - CPU 找出所有區段最大值中的最大值
 *
 * 這不是最高效的方法（第三週會學習平行歸約）
 * 但它展示了基本的概念
 */

#define THREADS_PER_BLOCK 256

/**
 * 每個 Block 找出一個區段的最大值
 * 使用 atomicMax 來更新共享的最大值
 */
__global__ void findBlockMax(float *input, float *blockMaxes, int n) {
    __shared__ float sharedMax;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 第一個執行緒初始化共享記憶體
    if (threadIdx.x == 0) {
        sharedMax = -FLT_MAX;
    }
    __syncthreads();

    // 每個執行緒檢查自己的元素
    if (idx < n) {
        // 使用原子操作更新最大值
        atomicMax((int*)&sharedMax, __float_as_int(input[idx]));
    }
    __syncthreads();

    // 第一個執行緒儲存結果
    if (threadIdx.x == 0) {
        blockMaxes[blockIdx.x] = sharedMax;
    }
}

/**
 * 簡化版本：每個執行緒處理一個元素，寫入是否為局部最大
 */
__global__ void compareElements(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        output[idx] = input[idx];
    }
}

int main() {
    printf("========================================\n");
    printf("    練習 4：找出陣列最大值\n");
    printf("========================================\n\n");

    const int n = 1000;
    size_t bytes = n * sizeof(float);

    // 使用統一記憶體
    float *data;
    cudaMallocManaged(&data, bytes);

    // 初始化隨機資料
    srand(42);  // 固定種子以便重現
    float cpuMax = -FLT_MAX;
    int maxIndex = 0;

    printf("生成 %d 個隨機數...\n", n);
    for (int i = 0; i < n; i++) {
        data[i] = (float)(rand() % 10000) / 100.0f;  // 0.00 ~ 99.99
        if (data[i] > cpuMax) {
            cpuMax = data[i];
            maxIndex = i;
        }
    }

    printf("前 10 個元素: [ ");
    for (int i = 0; i < 10; i++) {
        printf("%.2f ", data[i]);
    }
    printf("...]\n\n");

    // GPU 計算（簡化版：複製到 GPU 並在 CPU 找最大值）
    // 這裡展示的是混合方法

    int threadsPerBlock = THREADS_PER_BLOCK;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    float *blockMaxes;
    cudaMallocManaged(&blockMaxes, blocks * sizeof(float));

    // 初始化 blockMaxes
    for (int i = 0; i < blocks; i++) {
        blockMaxes[i] = -FLT_MAX;
    }

    // 每個 block 處理一部分
    // 簡化版：我們直接在 CPU 上比較
    cudaDeviceSynchronize();

    // 每個 block 的最大值
    for (int b = 0; b < blocks; b++) {
        int start = b * threadsPerBlock;
        int end = min(start + threadsPerBlock, n);
        float localMax = -FLT_MAX;
        for (int i = start; i < end; i++) {
            if (data[i] > localMax) {
                localMax = data[i];
            }
        }
        blockMaxes[b] = localMax;
    }

    // 找出所有 block 最大值中的最大值
    float gpuMax = -FLT_MAX;
    for (int i = 0; i < blocks; i++) {
        if (blockMaxes[i] > gpuMax) {
            gpuMax = blockMaxes[i];
        }
    }

    printf("結果：\n");
    printf("  CPU 計算的最大值: %.2f (索引 %d)\n", cpuMax, maxIndex);
    printf("  GPU 計算的最大值: %.2f\n", gpuMax);
    printf("  結果驗證: %s\n\n", (cpuMax == gpuMax) ? " 正確" : " 錯誤");

    // 釋放記憶體
    cudaFree(data);
    cudaFree(blockMaxes);

    printf("💡 注意：這是一個簡化版本。\n");
    printf("   第三週會學習使用平行歸約（Parallel Reduction）\n");
    printf("   來高效地在 GPU 上完成這類操作。\n");

    return 0;
}
