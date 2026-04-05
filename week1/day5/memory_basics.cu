#include <stdio.h>
#include <stdlib.h>

/**
 * 錯誤檢查巨集：包裝 CUDA API 呼叫，自動檢查是否有錯誤
 * 💡 Debug 提示：每個 CUDA API 呼叫都應該檢查回傳值，這個巨集讓你不用每次手寫
 */
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s (line %d)\n", \
               cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
}

/**
 * 簡單的 kernel 函式：將每個元素乘以 2
 */
// __global__ 標記此函式為 GPU kernel
__global__ void doubleArray(float *arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // 計算全域索引
    if (idx < n) {  // ⚠️ 注意：邊界檢查不可省略
        arr[idx] = arr[idx] * 2.0f;
    }
}

int main() {
    printf("========================================\n");
    printf("    CUDA Memory Management Basics\n");
    printf("========================================\n\n");

    const int n = 10;
    size_t bytes = n * sizeof(float);

    // ========== Host Memory ==========
    printf("1. Allocate host memory\n");
    float *h_data = (float*)malloc(bytes);
    if (h_data == NULL) {
        printf("Error: Cannot allocate host memory\n");
        return 1;
    }
    printf("   Allocated %zu bytes on CPU\n\n", bytes);

    // Initialize data
    printf("2. Initialize data\n");
    printf("   Original data: [ ");
    for (int i = 0; i < n; i++) {
        h_data[i] = (float)i;
        printf("%.1f ", h_data[i]);
    }
    printf("]\n\n");

    // ========== Device Memory ==========
    printf("3. Allocate device memory\n");
    float *d_data;  // 指向 GPU 記憶體的指標
    // cudaMalloc：在 GPU 上配置記憶體（CPU 不能直接存取這塊記憶體）
    CHECK_CUDA(cudaMalloc(&d_data, bytes));
    printf("   Allocated %zu bytes on GPU\n\n", bytes);

    // ========== Data Transfer: CPU -> GPU ==========
    printf("4. Copy data to GPU (cudaMemcpyHostToDevice)\n");
    // cudaMemcpy：CPU -> GPU 方向的資料複製（HostToDevice）
    // ⚠️ 注意：第一個參數是「目的地」，第二個是「來源」
    CHECK_CUDA(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));
    printf("   Transfer complete!\n\n");

    // ========== Execute Kernel ==========
    printf("5. Execute kernel (multiply each element by 2)\n");
    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    doubleArray<<<blocks, threadsPerBlock>>>(d_data, n);  // <<<grid大小, block大小>>> 啟動 kernel
    // cudaDeviceSynchronize：等待 GPU 完成所有工作
    CHECK_CUDA(cudaDeviceSynchronize());
    printf("   Kernel execution complete!\n\n");

    // ========== Data Transfer: GPU -> CPU ==========
    printf("6. Copy results back to CPU (cudaMemcpyDeviceToHost)\n");
    // cudaMemcpy：GPU -> CPU 方向的資料複製（DeviceToHost）
    CHECK_CUDA(cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost));
    printf("   Transfer complete!\n\n");

    // Display results
    printf("7. Results\n");
    printf("   After processing: [ ");
    for (int i = 0; i < n; i++) {
        printf("%.1f ", h_data[i]);
    }
    printf("]\n\n");

    // ========== Using cudaMemset ==========
    printf("8. Using cudaMemset to clear GPU memory\n");
    // cudaMemset：將 GPU 記憶體填入指定值（這裡填 0，類似 CPU 的 memset）
    CHECK_CUDA(cudaMemset(d_data, 0, bytes));
    CHECK_CUDA(cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost));
    printf("   After clearing: [ ");
    for (int i = 0; i < n; i++) {
        printf("%.1f ", h_data[i]);
    }
    printf("]\n\n");

    // ========== Free Memory ==========
    printf("9. Free memory\n");
    CHECK_CUDA(cudaFree(d_data));  // 釋放 GPU 記憶體
    free(h_data);  // 釋放 CPU 記憶體
    printf("   Memory freed\n");

    printf("\n========================================\n");
    printf(" Memory management demo complete!\n");
    printf("========================================\n");

    return 0;
}
