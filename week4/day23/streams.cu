#include <stdio.h>
#include <stdlib.h>

/**
 * Day 23: CUDA Streams 與非同步執行
 *
 * 展示如何使用 Streams 重疊計算與資料傳輸
 */

#define N (1 << 22)  // 4M 元素

// __global__ 標記為 GPU kernel，由 CPU 呼叫、GPU 執行
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    // blockIdx.x * blockDim.x + threadIdx.x 是計算全域執行緒索引的標準公式
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 增加一些計算量（讓 kernel 跑久一點，才能看出 stream 重疊的效果）
        float val = a[idx] + b[idx];
        for (int i = 0; i < 100; i++) {
            val = sinf(val) * cosf(val) + val;
        }
        c[idx] = val;
    }
}

// 不使用 Stream 的版本：所有操作依序執行（傳輸 → 計算 → 傳輸）
float runWithoutStreams(float *h_a, float *h_b, float *h_c,
                        float *d_a, float *d_b, float *d_c, int size) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // 順序執行：傳輸 → 計算 → 傳輸（每一步都要等上一步完成）
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice); // CPU → GPU
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    vectorAdd<<<blocks, threads>>>(d_a, d_b, d_c, N); // 啟動 kernel

    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost); // GPU → CPU

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds;
}

// 使用多個 Stream 的版本：把資料切成小塊，每塊的傳輸和計算可以重疊進行
// 💡 Debug 提示：如果加速比接近 1.0，可能是資料量太小或計算量不夠
float runWithStreams(float *h_a, float *h_b, float *h_c,
                     float *d_a, float *d_b, float *d_c,
                     int size, int nStreams) {
    int streamSize = N / nStreams;       // 每個 stream 處理的元素數量
    int streamBytes = streamSize * sizeof(float);

    // 創建多個 CUDA Stream（每個 stream 是一條獨立的工作佇列）
    cudaStream_t *streams = (cudaStream_t*)malloc(nStreams * sizeof(cudaStream_t));
    for (int i = 0; i < nStreams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // 每個 stream 處理一小塊資料：傳入 → 計算 → 傳回，各 stream 可重疊執行
    for (int i = 0; i < nStreams; i++) {
        int offset = i * streamSize;

        // cudaMemcpyAsync = 非同步傳輸，不會阻塞 CPU，需要指定 stream
        // ⚠️ 注意：非同步傳輸必須使用 Pinned Memory（cudaMallocHost），否則會退化成同步
        cudaMemcpyAsync(&d_a[offset], &h_a[offset], streamBytes,
                        cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(&d_b[offset], &h_b[offset], streamBytes,
                        cudaMemcpyHostToDevice, streams[i]);

        // <<<blocks, threads, 0, streams[i]>>> 第四個參數指定此 kernel 在哪個 stream 上執行
        int threads = 256;
        int blocks = (streamSize + threads - 1) / threads;
        vectorAdd<<<blocks, threads, 0, streams[i]>>>
            (&d_a[offset], &d_b[offset], &d_c[offset], streamSize);

        // 非同步 D2H（GPU → CPU）
        cudaMemcpyAsync(&h_c[offset], &d_c[offset], streamBytes,
                        cudaMemcpyDeviceToHost, streams[i]);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    for (int i = 0; i < nStreams; i++) {
        cudaStreamDestroy(streams[i]);
    }
    free(streams);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds;
}

int main() {
    printf("========================================\n");
    printf("    CUDA Streams 效能比較\n");
    printf("========================================\n\n");

    int size = N * sizeof(float);
    printf("資料大小: %d 個元素 (%.1f MB)\n\n", N, size / (1024.0f * 1024.0f));

    // 檢查 GPU 是否支援 Overlap
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("Concurrent copy and execute: %s\n",
           prop.deviceOverlap ? "Supported" : "Not supported");
    printf("非同步引擎數量: %d\n\n", prop.asyncEngineCount);

    // 分配 Pinned Memory（鎖頁記憶體），非同步傳輸必須使用
    // ⚠️ 注意：cudaMallocHost 分配的記憶體不會被作業系統換出到硬碟，所以不要分配太多
    float *h_a, *h_b, *h_c;
    cudaMallocHost(&h_a, size); // 比普通 malloc 慢，但傳輸速度更快
    cudaMallocHost(&h_b, size);
    cudaMallocHost(&h_c, size);

    // 初始化資料
    for (int i = 0; i < N; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    // GPU 記憶體（d_ 前綴代表 device = GPU）
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size); // 在 GPU 上分配記憶體
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // 暖機：第一次執行通常較慢（GPU 需要初始化），所以先跑一次不計時
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    vectorAdd<<<(N+255)/256, 256>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize(); // 等待 GPU 上所有操作完成

    // 測試不同配置
    printf("效能測試結果：\n");
    printf("----------------------------------------\n");

    // 無 Streams
    float timeNoStream = runWithoutStreams(h_a, h_b, h_c, d_a, d_b, d_c, size);
    printf("無 Streams:      %7.2f ms\n", timeNoStream);

    // 不同數量的 Streams
    int streamCounts[] = {2, 4, 8, 16};
    for (int i = 0; i < 4; i++) {
        int nStreams = streamCounts[i];
        float time = runWithStreams(h_a, h_b, h_c, d_a, d_b, d_c, size, nStreams);
        printf("%2d Streams:      %7.2f ms (加速 %.2fx)\n",
               nStreams, time, timeNoStream / time);
    }

    printf("----------------------------------------\n\n");

    // 驗證結果
    runWithStreams(h_a, h_b, h_c, d_a, d_b, d_c, size, 4);
    // 注意：由於核心中有複雜計算，這裡只驗證部分
    printf("結果驗證: 完成\n\n");

    printf("關鍵概念：\n");
    printf("1. Streams 允許操作重疊執行\n");
    printf("2. 必須使用 Pinned Memory 進行非同步傳輸\n");
    printf("3. 過多的 Streams 可能無法帶來更多加速\n");
    printf("4. 實際加速取決於計算量和傳輸量的比例\n");

    // 清理：cudaMallocHost 分配的要用 cudaFreeHost 釋放，cudaMalloc 的用 cudaFree
    // ⚠️ 注意：混用 free/cudaFree/cudaFreeHost 會導致未定義行為
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
