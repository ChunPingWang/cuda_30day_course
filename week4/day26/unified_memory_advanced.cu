#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/**
 * Day 26: Unified Memory 進階使用範例
 *
 * 展示預取和記憶體提示的效能影響
 */

#define N (1 << 24)  // 16M 元素

__global__ void processData(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = data[idx];
        // 一些計算
        for (int i = 0; i < 10; i++) {
            val = sinf(val) * cosf(val) + sqrtf(fabsf(val));
        }
        data[idx] = val;
    }
}

// 基本 Unified Memory（無優化）
float testBasicUM(int n) {
    float *data;
    cudaMallocManaged(&data, n * sizeof(float));

    // CPU 初始化
    for (int i = 0; i < n; i++) {
        data[i] = (float)i / n;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    // 暖機
    processData<<<blocks, threads>>>(data, n);
    cudaDeviceSynchronize();

    // 重新初始化
    for (int i = 0; i < n; i++) {
        data[i] = (float)i / n;
    }

    // 計時
    cudaEventRecord(start);
    processData<<<blocks, threads>>>(data, n);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    cudaFree(data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms;
}

// 使用預取優化
float testWithPrefetch(int n) {
    float *data;
    cudaMallocManaged(&data, n * sizeof(float));

    // CPU 初始化
    for (int i = 0; i < n; i++) {
        data[i] = (float)i / n;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    // 暖機
    cudaMemPrefetchAsync(data, n * sizeof(float), 0, NULL);
    processData<<<blocks, threads>>>(data, n);
    cudaDeviceSynchronize();

    // 預取回 CPU 進行重新初始化
    cudaMemPrefetchAsync(data, n * sizeof(float), cudaCpuDeviceId, NULL);
    cudaDeviceSynchronize();

    for (int i = 0; i < n; i++) {
        data[i] = (float)i / n;
    }

    // 計時（包含預取）
    cudaEventRecord(start);
    cudaMemPrefetchAsync(data, n * sizeof(float), 0, NULL);
    processData<<<blocks, threads>>>(data, n);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    cudaFree(data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms;
}

// 使用記憶體提示
float testWithHints(int n) {
    float *data;
    cudaMallocManaged(&data, n * sizeof(float));

    // 設定記憶體提示
    cudaMemAdvise(data, n * sizeof(float), cudaMemAdviseSetPreferredLocation, 0);
    cudaMemAdvise(data, n * sizeof(float), cudaMemAdviseSetAccessedBy, cudaCpuDeviceId);

    // CPU 初始化
    for (int i = 0; i < n; i++) {
        data[i] = (float)i / n;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    // 暖機
    processData<<<blocks, threads>>>(data, n);
    cudaDeviceSynchronize();

    // 重新初始化
    for (int i = 0; i < n; i++) {
        data[i] = (float)i / n;
    }

    // 計時
    cudaEventRecord(start);
    processData<<<blocks, threads>>>(data, n);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    cudaFree(data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms;
}

// 傳統方式（基準測試）
float testTraditional(int n) {
    float *h_data = (float*)malloc(n * sizeof(float));
    float *d_data;
    cudaMalloc(&d_data, n * sizeof(float));

    // CPU 初始化
    for (int i = 0; i < n; i++) {
        h_data[i] = (float)i / n;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    // 暖機
    cudaMemcpy(d_data, h_data, n * sizeof(float), cudaMemcpyHostToDevice);
    processData<<<blocks, threads>>>(d_data, n);
    cudaDeviceSynchronize();

    // 重新初始化
    for (int i = 0; i < n; i++) {
        h_data[i] = (float)i / n;
    }

    // 計時（包含傳輸）
    cudaEventRecord(start);
    cudaMemcpy(d_data, h_data, n * sizeof(float), cudaMemcpyHostToDevice);
    processData<<<blocks, threads>>>(d_data, n);
    cudaMemcpy(h_data, d_data, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    free(h_data);
    cudaFree(d_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms;
}

int main() {
    printf("========================================\n");
    printf("    Unified Memory 效能測試\n");
    printf("========================================\n\n");

    // 檢查 GPU 資訊
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("計算能力: %d.%d\n", prop.major, prop.minor);
    printf("Unified Memory Support: %s\n",
           prop.managedMemory ? "Yes" : "No");
    printf("Concurrent Managed Access: %s\n\n",
           prop.concurrentManagedAccess ? "Yes" : "No");

    int size = N * sizeof(float);
    printf("資料大小: %d 個元素 (%.1f MB)\n\n", N, size / (1024.0f * 1024.0f));

    printf("執行測試（多次平均）...\n\n");

    // 執行多次取平均
    int numRuns = 5;
    float timeBasic = 0, timePrefetch = 0, timeHints = 0, timeTraditional = 0;

    for (int i = 0; i < numRuns; i++) {
        timeBasic += testBasicUM(N);
        timePrefetch += testWithPrefetch(N);
        timeHints += testWithHints(N);
        timeTraditional += testTraditional(N);
    }

    timeBasic /= numRuns;
    timePrefetch /= numRuns;
    timeHints /= numRuns;
    timeTraditional /= numRuns;

    printf("結果：\n");
    printf("----------------------------------------\n");
    printf("方法                      時間(ms)\n");
    printf("----------------------------------------\n");
    printf("傳統 cudaMalloc+Memcpy    %7.3f\n", timeTraditional);
    printf("基本 Unified Memory       %7.3f\n", timeBasic);
    printf("UM + 預取                 %7.3f\n", timePrefetch);
    printf("UM + 記憶體提示           %7.3f\n", timeHints);
    printf("----------------------------------------\n\n");

    printf("與傳統方式比較：\n");
    printf("  基本 UM:     %.2fx\n", timeBasic / timeTraditional);
    printf("  UM + 預取:   %.2fx\n", timePrefetch / timeTraditional);
    printf("  UM + 提示:   %.2fx\n", timeHints / timeTraditional);
    printf("\n");

    printf("結論：\n");
    printf("1. 基本 Unified Memory 可能比傳統方式慢（頁面錯誤開銷）\n");
    printf("2. 使用預取可以顯著提升效能\n");
    printf("3. 記憶體提示在特定情況下有幫助\n");
    printf("4. 對於計算密集型任務，差異較小\n");

    return 0;
}
