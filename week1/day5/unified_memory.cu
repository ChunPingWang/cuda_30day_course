#include <stdio.h>

/**
 * 使用統一記憶體（Unified Memory）的簡化程式
 *
 * 優點：
 * - 不需要手動分配兩份記憶體
 * - 不需要手動複製資料
 * - 程式碼更簡潔
 *
 * 缺點：
 * - 效能可能稍差（自動傳輸開銷）
 * - 需要 CUDA 6.0+
 */

__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    printf("========================================\n");
    printf("    統一記憶體（Unified Memory）示範\n");
    printf("========================================\n\n");

    const int n = 10;
    size_t bytes = n * sizeof(float);

    // ========== 使用 cudaMallocManaged 分配統一記憶體 ==========
    printf("1. 使用 cudaMallocManaged 分配統一記憶體\n");

    float *a, *b, *c;
    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytes);

    printf("   已分配 %zu bytes × 3 的統一記憶體\n\n", bytes);

    // ========== 直接在 CPU 初始化（不需要 cudaMemcpy！）==========
    printf("2. 直接在 CPU 初始化資料\n");

    printf("   A = [ ");
    for (int i = 0; i < n; i++) {
        a[i] = (float)i;
        printf("%.0f ", a[i]);
    }
    printf("]\n");

    printf("   B = [ ");
    for (int i = 0; i < n; i++) {
        b[i] = (float)(i * 10);
        printf("%.0f ", b[i]);
    }
    printf("]\n\n");

    // ========== 執行核心函數 ==========
    printf("3. 執行 GPU 核心函數\n");

    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    vectorAdd<<<blocks, threadsPerBlock>>>(a, b, c, n);

    // 重要：等待 GPU 完成
    cudaDeviceSynchronize();

    printf("   核心函數執行完成！\n\n");

    // ========== 直接在 CPU 讀取結果（不需要 cudaMemcpy！）==========
    printf("4. 直接在 CPU 讀取結果\n");
    printf("   C = A + B = [ ");
    for (int i = 0; i < n; i++) {
        printf("%.0f ", c[i]);
    }
    printf("]\n\n");

    // 驗證結果
    printf("5. 驗證結果\n");
    bool correct = true;
    for (int i = 0; i < n; i++) {
        if (c[i] != a[i] + b[i]) {
            printf("    錯誤：c[%d] = %.0f, 預期 %.0f\n", i, c[i], a[i] + b[i]);
            correct = false;
        }
    }
    if (correct) {
        printf("    所有結果正確！\n\n");
    }

    // ========== 釋放記憶體 ==========
    printf("6. 釋放記憶體\n");
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    printf("   記憶體已釋放\n");

    // ========== 比較程式碼差異 ==========
    printf("\n========================================\n");
    printf("💡 程式碼比較：\n");
    printf("========================================\n\n");

    printf("傳統方式需要：\n");
    printf("  - malloc() 和 cudaMalloc() 分別分配\n");
    printf("  - cudaMemcpy(..., HostToDevice) 傳送資料\n");
    printf("  - cudaMemcpy(..., DeviceToHost) 取回結果\n");
    printf("  - free() 和 cudaFree() 分別釋放\n\n");

    printf("統一記憶體只需要：\n");
    printf("  - cudaMallocManaged() 一次分配\n");
    printf("  - cudaDeviceSynchronize() 確保完成\n");
    printf("  - cudaFree() 一次釋放\n");

    printf("\n========================================\n");
    printf(" 統一記憶體示範完成！\n");
    printf("========================================\n");

    return 0;
}
