#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 256

/**
 * 版本 1：基本歸約（交錯配對）
 */
__global__ void reduce_v1(float *input, float *output, int n) {
    __shared__ float sdata[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 載入資料到共享記憶體
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();

    // 歸約：交錯配對
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // 第一個執行緒寫入結果
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

/**
 * 版本 2：每個執行緒處理多個元素
 */
__global__ void reduce_v2(float *input, float *output, int n) {
    __shared__ float sdata[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // 每個執行緒載入兩個元素並相加
    float sum = 0.0f;
    if (idx < n) sum += input[idx];
    if (idx + blockDim.x < n) sum += input[idx + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();

    // 歸約
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

/**
 * 版本 3：Warp 層級優化
 */
__device__ void warpReduce(volatile float *sdata, int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

__global__ void reduce_v3(float *input, float *output, int n) {
    __shared__ float sdata[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    float sum = 0.0f;
    if (idx < n) sum += input[idx];
    if (idx + blockDim.x < n) sum += input[idx + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();

    // 歸約到 64 個元素
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Warp 內歸約（不需要同步）
    if (tid < 32) {
        warpReduce(sdata, tid);
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

/**
 * 完成最終歸約（在 CPU 或另一個核心函數中）
 */
float finalReduce(float *d_partial, int numBlocks) {
    float *h_partial = (float*)malloc(numBlocks * sizeof(float));
    cudaMemcpy(h_partial, d_partial, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);

    float sum = 0.0f;
    for (int i = 0; i < numBlocks; i++) {
        sum += h_partial[i];
    }

    free(h_partial);
    return sum;
}

int main() {
    printf("========================================\n");
    printf("    平行歸約（Reduction）示範\n");
    printf("========================================\n\n");

    const int N = 1 << 20;  // 約 100 萬個元素
    size_t bytes = N * sizeof(float);

    printf("陣列大小: %d 元素\n\n", N);

    // 分配記憶體
    float *h_input = (float*)malloc(bytes);
    float *d_input, *d_partial;

    cudaMalloc(&d_input, bytes);

    // 初始化（全部設為 1，這樣總和應該 = N）
    for (int i = 0; i < N; i++) {
        h_input[i] = 1.0f;
    }

    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms;

    // ========== 版本 1 ==========
    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cudaMalloc(&d_partial, blocks * sizeof(float));

    cudaEventRecord(start);
    reduce_v1<<<blocks, BLOCK_SIZE>>>(d_input, d_partial, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);

    float sum = finalReduce(d_partial, blocks);
    printf("版本 1（基本）:\n");
    printf("  時間: %.3f ms\n", ms);
    printf("  結果: %.0f（預期: %d）%s\n\n", sum, N,
           (int)sum == N ? "" : "");

    cudaFree(d_partial);

    // ========== 版本 2 ==========
    blocks = (N + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);
    cudaMalloc(&d_partial, blocks * sizeof(float));

    cudaEventRecord(start);
    reduce_v2<<<blocks, BLOCK_SIZE>>>(d_input, d_partial, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);

    sum = finalReduce(d_partial, blocks);
    printf("版本 2（每執行緒 2 元素）:\n");
    printf("  時間: %.3f ms\n", ms);
    printf("  結果: %.0f（預期: %d）%s\n\n", sum, N,
           (int)sum == N ? "" : "");

    cudaFree(d_partial);

    // ========== 版本 3 ==========
    cudaMalloc(&d_partial, blocks * sizeof(float));

    cudaEventRecord(start);
    reduce_v3<<<blocks, BLOCK_SIZE>>>(d_input, d_partial, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);

    sum = finalReduce(d_partial, blocks);
    printf("版本 3（Warp 優化）:\n");
    printf("  時間: %.3f ms\n", ms);
    printf("  結果: %.0f（預期: %d）%s\n\n", sum, N,
           (int)sum == N ? "" : "");

    // 清理
    cudaFree(d_input);
    cudaFree(d_partial);
    free(h_input);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("========================================\n");
    printf("歸約優化關鍵：\n");
    printf("1. 減少 __syncthreads 次數\n");
    printf("2. 每個執行緒處理多個元素\n");
    printf("3. Warp 內不需要同步\n");
    printf("========================================\n");

    return 0;
}
