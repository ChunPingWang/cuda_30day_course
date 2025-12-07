#include <stdio.h>

/**
 * 練習 1：向量平方
 * 計算 B[i] = A[i] * A[i]
 */

__global__ void vectorSquare(float *a, float *b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        b[idx] = a[idx] * a[idx];
    }
}

int main() {
    printf("========================================\n");
    printf("    練習 1：向量平方\n");
    printf("========================================\n\n");

    const int n = 10;
    size_t bytes = n * sizeof(float);

    // 使用統一記憶體
    float *a, *b;
    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);

    // 初始化
    printf("A = [ ");
    for (int i = 0; i < n; i++) {
        a[i] = (float)(i + 1);
        printf("%.0f ", a[i]);
    }
    printf("]\n\n");

    // 執行核心函數
    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    vectorSquare<<<blocks, threadsPerBlock>>>(a, b, n);
    cudaDeviceSynchronize();

    // 顯示結果
    printf("B = A² = [ ");
    for (int i = 0; i < n; i++) {
        printf("%.0f ", b[i]);
    }
    printf("]\n\n");

    // 驗證
    bool correct = true;
    for (int i = 0; i < n; i++) {
        if (b[i] != a[i] * a[i]) {
            correct = false;
            break;
        }
    }
    printf("結果驗證: %s\n", correct ? " 正確" : " 錯誤");

    // 釋放記憶體
    cudaFree(a);
    cudaFree(b);

    return 0;
}
