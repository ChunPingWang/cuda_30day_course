#include <stdio.h>

/**
 * 練習 2：向量內積（Dot Product）
 * 結果 = A[0]*B[0] + A[1]*B[1] + ... + A[n-1]*B[n-1]
 *
 * 簡化版本：
 * - GPU 負責逐元素相乘（平行計算）
 * - CPU 負責最後的加總
 * （第 3 週會學到更有效率的 Parallel Reduction 方法）
 */

// 逐元素相乘的 kernel：c[i] = a[i] * b[i]
__global__ void elementwiseMultiply(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

int main() {
    printf("========================================\n");
    printf("    Exercise 2: Vector Dot Product\n");
    printf("========================================\n\n");

    const int n = 8;
    size_t bytes = n * sizeof(float);

    // 使用統一記憶體
    float *a, *b, *c;
    cudaMallocManaged(&a, bytes);  // A 向量
    cudaMallocManaged(&b, bytes);  // B 向量
    cudaMallocManaged(&c, bytes);  // 存放逐元素相乘的結果

    // Initialize
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

    // GPU：平行計算逐元素乘積
    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    elementwiseMultiply<<<blocks, threadsPerBlock>>>(a, b, c, n);
    cudaDeviceSynchronize();  // ⚠️ 注意：一定要等 GPU 算完才能在 CPU 做加總

    printf("A * B (element-wise) = [ ");
    for (int i = 0; i < n; i++) {
        printf("%.0f ", c[i]);
    }
    printf("]\n\n");

    // CPU：將所有乘積加總得到內積結果
    // 💡 Debug 提示：這裡的加總是在 CPU 上做的，資料量大時會是效能瓶頸
    float dotProduct = 0.0f;
    for (int i = 0; i < n; i++) {
        dotProduct += c[i];
    }

    printf("Dot Product = ");
    for (int i = 0; i < n; i++) {
        printf("%.0f", a[i] * b[i]);
        if (i < n - 1) printf(" + ");
    }
    printf("\n");
    printf("           = %.0f\n\n", dotProduct);

    // Verify
    float expected = 0.0f;
    for (int i = 0; i < n; i++) {
        expected += (i + 1) * (i + 1);  // 1^2 + 2^2 + 3^2 + ...
    }
    printf("Expected result: %.0f\n", expected);
    printf("Result verification: %s\n", (dotProduct == expected) ? "CORRECT" : "ERROR");

    // 釋放統一記憶體
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    printf("\nNote: This version performs summation on CPU.\n");
    printf("In Week 3, we will learn how to efficiently perform\n");
    printf("reduction operations on GPU.\n");

    return 0;
}
