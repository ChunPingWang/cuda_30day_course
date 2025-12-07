#include <stdio.h>

/**
 * 傳統方式：每個執行緒處理一個元素
 */
__global__ void traditionalProcess(float *arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        arr[idx] = arr[idx] * 2.0f;
    }
}

/**
 * Grid-Stride Loop：每個執行緒處理多個元素
 */
__global__ void gridStrideProcess(float *arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;  // 總執行緒數

    // 每個執行緒用步幅方式處理多個元素
    for (int i = idx; i < n; i += stride) {
        arr[i] = arr[i] * 2.0f;
    }
}

/**
 * 展示 Grid-Stride Loop 的運作方式
 */
__global__ void showStridePattern(int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // 只讓前幾個執行緒印出，避免輸出太多
    if (idx < 4) {
        printf("Thread %d 處理的元素: ", idx);
        for (int i = idx; i < n && i < idx + stride * 3; i += stride) {
            printf("%d ", i);
        }
        printf("...\n");
    }
}

int main() {
    printf("========================================\n");
    printf("    Grid-Stride Loop 模式示範\n");
    printf("========================================\n\n");

    const int n = 100;
    size_t bytes = n * sizeof(float);

    // 分配統一記憶體
    float *data;
    cudaMallocManaged(&data, bytes);

    // ========== 示範 1：展示步幅模式 ==========
    printf("示範 1: 步幅模式視覺化\n");
    printf("配置: <<<2, 4>>> (8 個執行緒處理 %d 個元素)\n", n);
    printf("每個執行緒處理 %d 個元素\n\n", n / 8);

    showStridePattern<<<2, 4>>>(n);
    cudaDeviceSynchronize();

    // ========== 示範 2：傳統方式 ==========
    printf("\n示範 2: 傳統方式\n");
    printf("需要 <<<(n+255)/256, 256>>> = <<<1, 256>>> 個執行緒\n");

    // 初始化資料
    for (int i = 0; i < n; i++) {
        data[i] = (float)i;
    }

    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    traditionalProcess<<<blocks, threadsPerBlock>>>(data, n);
    cudaDeviceSynchronize();

    printf("結果（前 10 個元素）: ");
    for (int i = 0; i < 10; i++) {
        printf("%.0f ", data[i]);
    }
    printf("...\n");

    // ========== 示範 3：Grid-Stride Loop ==========
    printf("\n示範 3: Grid-Stride Loop\n");
    printf("使用固定 <<<2, 256>>> 個執行緒處理 %d 個元素\n", n);

    // 重新初始化資料
    for (int i = 0; i < n; i++) {
        data[i] = (float)i;
    }

    gridStrideProcess<<<2, 256>>>(data, n);
    cudaDeviceSynchronize();

    printf("結果（前 10 個元素）: ");
    for (int i = 0; i < 10; i++) {
        printf("%.0f ", data[i]);
    }
    printf("...\n");

    // ========== 比較 ==========
    printf("\n========================================\n");
    printf("💡 比較：\n");
    printf("========================================\n\n");

    printf("傳統方式：\n");
    printf("  - 需要根據 n 調整 block 數量\n");
    printf("  - n 很大時，block 數量可能超過限制\n\n");

    printf("Grid-Stride Loop：\n");
    printf("  - 固定的執行緒配置\n");
    printf("  - 適用於任何大小的資料\n");
    printf("  - 更好的可重用性\n");
    printf("  - 更好的記憶體存取模式\n");

    // 釋放記憶體
    cudaFree(data);

    printf("\n========================================\n");
    printf(" Grid-Stride Loop 示範完成！\n");
    printf("========================================\n");

    return 0;
}
