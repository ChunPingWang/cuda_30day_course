#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 256

/**
 * 版本 1：基本歸約（交錯配對）
 * 將整個陣列的元素加總 — 每個 block 先算出局部總和，最後再合併
 */
__global__ void reduce_v1(float *input, float *output, int n) {
    // __shared__ 宣告共享記憶體：同一 block 內所有執行緒共享，速度比全域記憶體快很多
    __shared__ float sdata[BLOCK_SIZE];

    int tid = threadIdx.x;  // 區塊內的執行緒索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // 全域索引

    // 每個執行緒載入一個元素到共享記憶體（超出範圍的補 0）
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();  // ⚠️ 注意：必須等所有執行緒都載入完畢才能開始歸約

    // 歸約：每一輪將步長 s 減半，活躍執行緒數也減半
    // 例如 s=128: tid 0~127 分別加上 sdata[tid+128]
    //      s=64:  tid 0~63  分別加上 sdata[tid+64] ...以此類推
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();  // 每一輪都要同步
    }

    // 歸約完成後，sdata[0] 就是整個 block 的總和
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];  // 每個 block 輸出一個局部總和
    }
}

/**
 * 版本 2：每個執行緒處理兩個元素（減少 block 數量，提高效率）
 */
__global__ void reduce_v2(float *input, float *output, int n) {
    __shared__ float sdata[BLOCK_SIZE];

    int tid = threadIdx.x;
    // 注意 blockDim.x * 2：每個 block 負責的範圍是原來的兩倍
    int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // 每個執行緒先把兩個元素加起來再存入共享記憶體 → 第一輪歸約免費完成
    float sum = 0.0f;
    if (idx < n) sum += input[idx];
    if (idx + blockDim.x < n) sum += input[idx + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();

    // 之後的歸約步驟與版本 1 相同
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
 * 版本 3：Warp 層級優化 — 最後 32 個元素不需要 __syncthreads
 */
// __device__ 表示此函式只在 GPU 上執行，由其他 GPU 函式呼叫
// volatile 關鍵字確保編譯器不會優化掉對共享記憶體的讀寫（Warp 內隱式同步需要）
__device__ void warpReduce(volatile float *sdata, int tid) {
    // 同一個 Warp 內的 32 個執行緒天然同步（SIMT 架構），不需要 __syncthreads
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

    // 先用 __syncthreads 歸約到剩 64 個元素（s > 32 時還跨 Warp，需要同步）
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // 最後 32 個元素在同一個 Warp 內，不需要 __syncthreads → 省下大量同步開銷
    if (tid < 32) {
        warpReduce(sdata, tid);
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

/**
 * 完成最終歸約：將每個 block 的局部總和搬回 CPU 做最後加總
 * 💡 Debug 提示：如果 block 數量很多，也可以再啟動一個 kernel 在 GPU 上歸約
 */
float finalReduce(float *d_partial, int numBlocks) {
    float *h_partial = (float*)malloc(numBlocks * sizeof(float));
    cudaMemcpy(h_partial, d_partial, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);  // GPU → CPU

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

    cudaMalloc(&d_input, bytes);  // 在 GPU 上分配記憶體

    // 初始化（全部設為 1，這樣總和應該 = N，方便驗證正確性）
    for (int i = 0; i < N; i++) {
        h_input[i] = 1.0f;
    }

    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);  // CPU → GPU

    // CUDA 事件用於精確計時（比 CPU 計時更準確，因為 GPU 是非同步執行的）
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
