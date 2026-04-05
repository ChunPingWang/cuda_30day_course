#include <stdio.h>

/**
 * 傳統做法：每個執行緒只處理一個元素
 * 需要啟動足夠多的執行緒來覆蓋所有元素
 */
__global__ void traditionalProcess(float *arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        arr[idx] = arr[idx] * 2.0f;
    }
}

/**
 * Grid-Stride Loop：每個執行緒用跨步迴圈處理多個元素
 * 好處：不管資料多大，都可以用固定數量的執行緒處理
 */
__global__ void gridStrideProcess(float *arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;   // 起始索引
    // gridDim.x = grid 中有幾個 block
    int stride = blockDim.x * gridDim.x;  // 跨步大小 = 全部執行緒的總數

    // 每個執行緒從自己的 idx 開始，每次跳 stride 個位置，處理多個元素
    // 💡 Debug 提示：如果結果有部分元素沒被處理，檢查 stride 的計算是否正確
    for (int i = idx; i < n; i += stride) {
        arr[i] = arr[i] * 2.0f;
    }
}

/**
 * 視覺化展示 Grid-Stride Loop 的工作方式：印出每個執行緒負責哪些元素
 */
__global__ void showStridePattern(int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;  // 所有執行緒的總數

    // Only let first few threads print to avoid too much output
    if (idx < 4) {
        printf("Thread %d processes elements: ", idx);
        for (int i = idx; i < n && i < idx + stride * 3; i += stride) {
            printf("%d ", i);
        }
        printf("...\n");
    }
}

int main() {
    printf("========================================\n");
    printf("    Grid-Stride Loop Pattern Demo\n");
    printf("========================================\n\n");

    const int n = 100;
    size_t bytes = n * sizeof(float);

    // Allocate unified memory
    float *data;
    cudaMallocManaged(&data, bytes);  // 配置統一記憶體

    // ========== Demo 1: Show Stride Pattern ==========
    printf("Demo 1: Stride Pattern Visualization\n");
    printf("Config: <<<2, 4>>> (8 threads processing %d elements)\n", n);
    printf("Each thread processes %d elements\n\n", n / 8);

    // <<<2, 4>>> 表示 2 個 block，每個 block 4 個執行緒，共 8 個執行緒
    showStridePattern<<<2, 4>>>(n);
    cudaDeviceSynchronize();  // 等待 GPU 完成

    // ========== Demo 2: Traditional Approach ==========
    printf("\nDemo 2: Traditional Approach\n");
    printf("Needs <<<(n+255)/256, 256>>> = <<<1, 256>>> threads\n");

    // Initialize data
    for (int i = 0; i < n; i++) {
        data[i] = (float)i;
    }

    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    traditionalProcess<<<blocks, threadsPerBlock>>>(data, n);
    cudaDeviceSynchronize();

    printf("Results (first 10 elements): ");
    for (int i = 0; i < 10; i++) {
        printf("%.0f ", data[i]);
    }
    printf("...\n");

    // ========== Demo 3: Grid-Stride Loop ==========
    printf("\nDemo 3: Grid-Stride Loop\n");
    printf("Using fixed <<<2, 256>>> threads to process %d elements\n", n);

    // Reinitialize data
    for (int i = 0; i < n; i++) {
        data[i] = (float)i;
    }

    // 只用 2 個 block x 256 個執行緒 = 512 個執行緒，就能處理任意大小的陣列
    gridStrideProcess<<<2, 256>>>(data, n);
    cudaDeviceSynchronize();

    printf("Results (first 10 elements): ");
    for (int i = 0; i < 10; i++) {
        printf("%.0f ", data[i]);
    }
    printf("...\n");

    // ========== Comparison ==========
    printf("\n========================================\n");
    printf("Comparison:\n");
    printf("========================================\n\n");

    printf("Traditional approach:\n");
    printf("  - Need to adjust block count based on n\n");
    printf("  - Block count may exceed limit for large n\n\n");

    printf("Grid-Stride Loop:\n");
    printf("  - Fixed thread configuration\n");
    printf("  - Works for any data size\n");
    printf("  - Better reusability\n");
    printf("  - Better memory access patterns\n");

    // Free memory
    cudaFree(data);

    printf("\n========================================\n");
    printf(" Grid-Stride Loop demo complete!\n");
    printf("========================================\n");

    return 0;
}
