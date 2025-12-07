#include <stdio.h>
#include <stdlib.h>

/**
 * 錯誤檢查巨集
 */
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA 錯誤: %s (行 %d)\n", \
               cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
}

/**
 * 簡單的核心函數：將陣列每個元素乘以 2
 */
__global__ void doubleArray(float *arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        arr[idx] = arr[idx] * 2.0f;
    }
}

int main() {
    printf("========================================\n");
    printf("    CUDA 記憶體管理基礎示範\n");
    printf("========================================\n\n");

    const int n = 10;
    size_t bytes = n * sizeof(float);

    // ========== 主機端記憶體 ==========
    printf("1. 分配主機端記憶體\n");
    float *h_data = (float*)malloc(bytes);
    if (h_data == NULL) {
        printf("錯誤：無法分配主機端記憶體\n");
        return 1;
    }
    printf("   已分配 %zu bytes 在 CPU\n\n", bytes);

    // 初始化資料
    printf("2. 初始化資料\n");
    printf("   原始資料: [ ");
    for (int i = 0; i < n; i++) {
        h_data[i] = (float)i;
        printf("%.1f ", h_data[i]);
    }
    printf("]\n\n");

    // ========== 設備端記憶體 ==========
    printf("3. 分配設備端記憶體\n");
    float *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, bytes));
    printf("   已分配 %zu bytes 在 GPU\n\n", bytes);

    // ========== 資料傳輸：CPU → GPU ==========
    printf("4. 複製資料到 GPU (cudaMemcpyHostToDevice)\n");
    CHECK_CUDA(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));
    printf("   傳輸完成！\n\n");

    // ========== 執行核心函數 ==========
    printf("5. 執行核心函數（將每個元素乘以 2）\n");
    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    doubleArray<<<blocks, threadsPerBlock>>>(d_data, n);
    CHECK_CUDA(cudaDeviceSynchronize());
    printf("   核心函數執行完成！\n\n");

    // ========== 資料傳輸：GPU → CPU ==========
    printf("6. 複製結果回 CPU (cudaMemcpyDeviceToHost)\n");
    CHECK_CUDA(cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost));
    printf("   傳輸完成！\n\n");

    // 顯示結果
    printf("7. 結果\n");
    printf("   處理後: [ ");
    for (int i = 0; i < n; i++) {
        printf("%.1f ", h_data[i]);
    }
    printf("]\n\n");

    // ========== 使用 cudaMemset ==========
    printf("8. 使用 cudaMemset 清空 GPU 記憶體\n");
    CHECK_CUDA(cudaMemset(d_data, 0, bytes));
    CHECK_CUDA(cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost));
    printf("   清空後: [ ");
    for (int i = 0; i < n; i++) {
        printf("%.1f ", h_data[i]);
    }
    printf("]\n\n");

    // ========== 釋放記憶體 ==========
    printf("9. 釋放記憶體\n");
    CHECK_CUDA(cudaFree(d_data));
    free(h_data);
    printf("   記憶體已釋放\n");

    printf("\n========================================\n");
    printf(" 記憶體管理示範完成！\n");
    printf("========================================\n");

    return 0;
}
