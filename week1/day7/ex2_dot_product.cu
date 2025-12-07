#include <stdio.h>

/**
 * 練習 2：向量點積
 * result = A[0]*B[0] + A[1]*B[1] + ... + A[n-1]*B[n-1]
 *
 * 這是一個簡化版本：
 * - GPU 計算每個元素的乘積
 * - CPU 進行最終加總
 * （更高效的方法會在第三週學習：平行歸約）
 */

__global__ void elementwiseMultiply(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

int main() {
    printf("========================================\n");
    printf("    練習 2：向量點積\n");
    printf("========================================\n\n");

    const int n = 8;
    size_t bytes = n * sizeof(float);

    // 使用統一記憶體
    float *a, *b, *c;
    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytes);

    // 初始化
    printf("A = [ ");
    for (int i = 0; i < n; i++) {
        a[i] = (float)(i + 1);
        printf("%.0f ", a[i]);
    }
    printf("]\n");

    printf("B = [ ");
    for (int i = 0; i < n; i++) {
        b[i] = (float)(i + 1);
        printf("%.0f ", b[i]);
    }
    printf("]\n\n");

    // GPU: 計算元素乘積
    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    elementwiseMultiply<<<blocks, threadsPerBlock>>>(a, b, c, n);
    cudaDeviceSynchronize();

    printf("A * B（元素乘積）= [ ");
    for (int i = 0; i < n; i++) {
        printf("%.0f ", c[i]);
    }
    printf("]\n\n");

    // CPU: 加總
    float dotProduct = 0.0f;
    for (int i = 0; i < n; i++) {
        dotProduct += c[i];
    }

    printf("點積 = ");
    for (int i = 0; i < n; i++) {
        printf("%.0f", a[i] * b[i]);
        if (i < n - 1) printf(" + ");
    }
    printf("\n");
    printf("     = %.0f\n\n", dotProduct);

    // 驗證
    float expected = 0.0f;
    for (int i = 0; i < n; i++) {
        expected += (i + 1) * (i + 1);  // 1² + 2² + 3² + ...
    }
    printf("預期結果: %.0f\n", expected);
    printf("結果驗證: %s\n", (dotProduct == expected) ? " 正確" : " 錯誤");

    // 釋放記憶體
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    printf("\n💡 注意：這個版本在 CPU 上進行加總。\n");
    printf("   第三週我們會學習如何在 GPU 上高效地進行歸約操作。\n");

    return 0;
}
