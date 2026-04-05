#include <stdio.h>

/**
 * 練習 1：向量平方
 * 計算 B[i] = A[i] * A[i]，每個執行緒負責一個元素
 */

// __global__ 標記為 GPU kernel 函式
__global__ void vectorSquare(float *a, float *b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // 全域執行緒索引

    // ⚠️ 注意：邊界檢查，避免存取超出陣列的記憶體
    if (idx < n) {
        b[idx] = a[idx] * a[idx];
    }
}

int main() {
    printf("========================================\n");
    printf("    Exercise 1: Vector Square\n");
    printf("========================================\n\n");

    const int n = 10;
    size_t bytes = n * sizeof(float);

    // 使用統一記憶體（Unified Memory），CPU 和 GPU 共用同一個指標
    float *a, *b;
    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);

    // Initialize
    printf("A = [ ");
    for (int i = 0; i < n; i++) {
        a[i] = (float)(i + 1);
        printf("%.0f ", a[i]);
    }
    printf("]\n\n");

    // 啟動 kernel
    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;  // 無條件進位

    vectorSquare<<<blocks, threadsPerBlock>>>(a, b, n);  // <<<grid大小, block大小>>>
    cudaDeviceSynchronize();  // 等待 GPU 完成後才能在 CPU 端讀取結果

    // Display results
    printf("B = A^2 = [ ");
    for (int i = 0; i < n; i++) {
        printf("%.0f ", b[i]);
    }
    printf("]\n\n");

    // Verify
    bool correct = true;
    for (int i = 0; i < n; i++) {
        if (b[i] != a[i] * a[i]) {
            correct = false;
            break;
        }
    }
    printf("Result verification: %s\n", correct ? "CORRECT" : "ERROR");

    // 釋放統一記憶體
    cudaFree(a);
    cudaFree(b);

    return 0;
}
