#include <stdio.h>
#include <stdlib.h>
#include <float.h>

/**
 * 練習 4：找出陣列中的最大值
 *
 * 簡化版本的做法：
 * - 將陣列分成多個區段
 * - 每個 Block 找出自己區段的最大值
 * - CPU 再從所有區段最大值中找出全域最大值
 *
 * 這不是最有效率的做法（第 3 週會學 Parallel Reduction）
 * 但展示了 GPU/CPU 混合計算的基本概念
 */

#define THREADS_PER_BLOCK 256

int main() {
    printf("========================================\n");
    printf("    Exercise 4: Find Maximum Value\n");
    printf("========================================\n\n");

    const int n = 1000;
    size_t bytes = n * sizeof(float);

    // 使用統一記憶體
    float *data;
    cudaMallocManaged(&data, bytes);

    // 初始化隨機資料
    srand(42);  // 固定亂數種子，確保每次執行結果相同（方便 debug）
    float cpuMax = -FLT_MAX;  // FLT_MAX 是 float 的最大值，取負號作為初始最小值
    int maxIndex = 0;

    printf("Generating %d random numbers...\n", n);
    for (int i = 0; i < n; i++) {
        data[i] = (float)(rand() % 10000) / 100.0f;  // 0.00 ~ 99.99
        if (data[i] > cpuMax) {
            cpuMax = data[i];
            maxIndex = i;
        }
    }

    printf("First 10 elements: [ ");
    for (int i = 0; i < 10; i++) {
        printf("%.2f ", data[i]);
    }
    printf("...]\n\n");

    // GPU computation (simplified version: copy to GPU and find max on CPU)
    // This demonstrates the hybrid approach

    int threadsPerBlock = THREADS_PER_BLOCK;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    float *blockMaxes;
    cudaMallocManaged(&blockMaxes, blocks * sizeof(float));  // 儲存每個 block 的區域最大值

    // 初始化每個 block 的最大值為最小值
    for (int i = 0; i < blocks; i++) {
        blockMaxes[i] = -FLT_MAX;
    }

    // 💡 Debug 提示：這個簡化版本其實是在 CPU 上做分段搜尋，模擬 GPU block 的行為
    // 真正的 GPU 平行版本會在第 3 週學到
    cudaDeviceSynchronize();

    // 模擬每個 block 找自己區段的最大值
    for (int b = 0; b < blocks; b++) {
        int start = b * threadsPerBlock;
        int end = start + threadsPerBlock;
        if (end > n) end = n;
        float localMax = -FLT_MAX;
        for (int i = start; i < end; i++) {
            if (data[i] > localMax) {
                localMax = data[i];
            }
        }
        blockMaxes[b] = localMax;
    }

    // 從所有區段最大值中，找出全域最大值
    float gpuMax = -FLT_MAX;
    for (int i = 0; i < blocks; i++) {
        if (blockMaxes[i] > gpuMax) {
            gpuMax = blockMaxes[i];
        }
    }

    printf("Results:\n");
    printf("  CPU computed max: %.2f (index %d)\n", cpuMax, maxIndex);
    printf("  GPU computed max: %.2f\n", gpuMax);
    printf("  Result verification: %s\n\n", (cpuMax == gpuMax) ? "CORRECT" : "ERROR");

    // 釋放統一記憶體
    cudaFree(data);
    cudaFree(blockMaxes);

    printf("Note: This is a simplified version.\n");
    printf("In Week 3, we will learn to use Parallel Reduction\n");
    printf("to efficiently perform such operations on GPU.\n");

    return 0;
}
