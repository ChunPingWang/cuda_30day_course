#include <stdio.h>
#include <stdlib.h>

/**
 * Day 24: 多 GPU 程式設計範例
 *
 * 展示如何查詢和使用多個 GPU
 */

#define N (1 << 22)  // 4M 元素

__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

void queryDevices() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    printf("========================================\n");
    printf("    系統 GPU 資訊\n");
    printf("========================================\n\n");

    printf("找到 %d 個 CUDA 裝置\n\n", deviceCount);

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        printf("裝置 %d: %s\n", i, prop.name);
        printf("  計算能力: %d.%d\n", prop.major, prop.minor);
        printf("  全域記憶體: %.1f GB\n", prop.totalGlobalMem / 1e9);
        printf("  SM 數量: %d\n", prop.multiProcessorCount);
        printf("  每個 Block 最大執行緒數: %d\n", prop.maxThreadsPerBlock);
        printf("  Warp 大小: %d\n", prop.warpSize);

        // 檢查 P2P 支援
        for (int j = 0; j < deviceCount; j++) {
            if (i != j) {
                int canAccess;
                cudaDeviceCanAccessPeer(&canAccess, i, j);
                printf("  P2P 到裝置 %d: %s\n", j, canAccess ? "支援" : "不支援");
            }
        }
        printf("\n");
    }
}

float runSingleGPU(float *h_a, float *h_b, float *h_c, int n) {
    float *d_a, *d_b, *d_c;
    int size = n * sizeof(float);

    cudaSetDevice(0);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaEventRecord(start);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    vectorAdd<<<blocks, threads>>>(d_a, d_b, d_c, n);

    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms;
}

float runMultiGPU(float *h_a, float *h_b, float *h_c, int n, int numGPUs) {
    int size = n * sizeof(float);
    int chunkSize = n / numGPUs;
    int chunkBytes = chunkSize * sizeof(float);

    float **d_a = (float**)malloc(numGPUs * sizeof(float*));
    float **d_b = (float**)malloc(numGPUs * sizeof(float*));
    float **d_c = (float**)malloc(numGPUs * sizeof(float*));
    cudaStream_t *streams = (cudaStream_t*)malloc(numGPUs * sizeof(cudaStream_t));

    // 在每個 GPU 上分配記憶體和創建 stream
    for (int i = 0; i < numGPUs; i++) {
        cudaSetDevice(i);
        cudaMalloc(&d_a[i], chunkBytes);
        cudaMalloc(&d_b[i], chunkBytes);
        cudaMalloc(&d_c[i], chunkBytes);
        cudaStreamCreate(&streams[i]);
    }

    cudaEvent_t start, stop;
    cudaSetDevice(0);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // 在每個 GPU 上平行執行
    for (int i = 0; i < numGPUs; i++) {
        cudaSetDevice(i);
        int offset = i * chunkSize;

        cudaMemcpyAsync(d_a[i], h_a + offset, chunkBytes,
                        cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(d_b[i], h_b + offset, chunkBytes,
                        cudaMemcpyHostToDevice, streams[i]);

        int threads = 256;
        int blocks = (chunkSize + threads - 1) / threads;
        vectorAdd<<<blocks, threads, 0, streams[i]>>>(d_a[i], d_b[i], d_c[i], chunkSize);

        cudaMemcpyAsync(h_c + offset, d_c[i], chunkBytes,
                        cudaMemcpyDeviceToHost, streams[i]);
    }

    // 同步所有 GPU
    for (int i = 0; i < numGPUs; i++) {
        cudaSetDevice(i);
        cudaStreamSynchronize(streams[i]);
    }

    cudaSetDevice(0);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    // 清理
    for (int i = 0; i < numGPUs; i++) {
        cudaSetDevice(i);
        cudaFree(d_a[i]);
        cudaFree(d_b[i]);
        cudaFree(d_c[i]);
        cudaStreamDestroy(streams[i]);
    }

    free(d_a);
    free(d_b);
    free(d_c);
    free(streams);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms;
}

int main() {
    // 查詢裝置
    queryDevices();

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount < 1) {
        printf("沒有找到 CUDA 裝置！\n");
        return 1;
    }

    // 分配主機記憶體（pinned）
    float *h_a, *h_b, *h_c;
    int size = N * sizeof(float);

    cudaMallocHost(&h_a, size);
    cudaMallocHost(&h_b, size);
    cudaMallocHost(&h_c, size);

    // 初始化
    for (int i = 0; i < N; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    printf("========================================\n");
    printf("    效能測試\n");
    printf("========================================\n\n");
    printf("資料大小: %d 個元素 (%.1f MB)\n\n", N, size / (1024.0f * 1024.0f));

    // 單 GPU 測試
    float timeSingle = runSingleGPU(h_a, h_b, h_c, N);
    printf("單 GPU 執行時間: %.3f ms\n", timeSingle);

    // 驗證
    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (h_c[i] != 3.0f) {
            correct = false;
            break;
        }
    }
    printf("驗證: %s\n\n", correct ? "通過" : "失敗");

    // 多 GPU 測試（如果有多個 GPU）
    if (deviceCount > 1) {
        float timeMulti = runMultiGPU(h_a, h_b, h_c, N, deviceCount);
        printf("%d GPU 執行時間: %.3f ms\n", deviceCount, timeMulti);

        // 驗證
        correct = true;
        for (int i = 0; i < N; i++) {
            if (h_c[i] != 3.0f) {
                correct = false;
                break;
            }
        }
        printf("驗證: %s\n", correct ? "通過" : "失敗");
        printf("加速比: %.2fx\n", timeSingle / timeMulti);
    } else {
        printf("只有一個 GPU，跳過多 GPU 測試\n");
    }

    // 清理
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);

    return 0;
}
