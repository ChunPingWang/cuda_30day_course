#include <stdio.h>

/**
 * Using Unified Memory - Simplified program
 *
 * Advantages:
 * - No need to manually allocate two copies of memory
 * - No need to manually copy data
 * - Cleaner code
 *
 * Disadvantages:
 * - Performance may be slightly lower (automatic transfer overhead)
 * - Requires CUDA 6.0+
 */

// 向量加法 kernel：和一般版本完全一樣，kernel 本身不需要知道記憶體是統一記憶體還是傳統記憶體
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // 全域索引
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    printf("========================================\n");
    printf("    Unified Memory Demo\n");
    printf("========================================\n\n");

    const int n = 10;
    size_t bytes = n * sizeof(float);

    // ========== Allocate unified memory using cudaMallocManaged ==========
    printf("1. Allocate unified memory using cudaMallocManaged\n");

    float *a, *b, *c;
    // cudaMallocManaged：配置「統一記憶體」，CPU 和 GPU 都可以直接存取同一個指標
    // 不再需要分別 malloc + cudaMalloc，也不需要 cudaMemcpy 手動搬資料
    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytes);

    printf("   Allocated %zu bytes x 3 of unified memory\n\n", bytes);

    // ========== Initialize directly on CPU (no cudaMemcpy needed!) ==========
    printf("2. Initialize data directly on CPU\n");

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

    // ========== Execute Kernel ==========
    printf("3. Execute GPU kernel\n");

    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    // <<<>>> 啟動 kernel，統一記憶體的指標可以直接傳入
    vectorAdd<<<blocks, threadsPerBlock>>>(a, b, c, n);

    // ⚠️ 注意：使用統一記憶體時，一定要先 cudaDeviceSynchronize() 才能在 CPU 讀取結果
    // 否則 GPU 可能還沒算完，CPU 就去讀取了（會得到錯誤的結果）
    cudaDeviceSynchronize();

    printf("   Kernel execution complete!\n\n");

    // ========== Read results directly on CPU (no cudaMemcpy needed!) ==========
    printf("4. Read results directly on CPU\n");
    printf("   C = A + B = [ ");
    for (int i = 0; i < n; i++) {
        printf("%.0f ", c[i]);
    }
    printf("]\n\n");

    // Verify results
    printf("5. Verify results\n");
    bool correct = true;
    for (int i = 0; i < n; i++) {
        if (c[i] != a[i] + b[i]) {
            printf("    Error: c[%d] = %.0f, expected %.0f\n", i, c[i], a[i] + b[i]);
            correct = false;
        }
    }
    if (correct) {
        printf("    All results correct!\n\n");
    }

    // ========== Free Memory ==========
    printf("6. Free memory\n");
    // 統一記憶體用 cudaFree 釋放（不需要額外的 free）
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    printf("   Memory freed\n");

    // ========== Code Comparison ==========
    printf("\n========================================\n");
    printf("Code Comparison:\n");
    printf("========================================\n\n");

    printf("Traditional approach requires:\n");
    printf("  - malloc() and cudaMalloc() for separate allocation\n");
    printf("  - cudaMemcpy(..., HostToDevice) to send data\n");
    printf("  - cudaMemcpy(..., DeviceToHost) to get results\n");
    printf("  - free() and cudaFree() for separate freeing\n\n");

    printf("Unified memory only needs:\n");
    printf("  - cudaMallocManaged() for single allocation\n");
    printf("  - cudaDeviceSynchronize() to ensure completion\n");
    printf("  - cudaFree() for single freeing\n");

    printf("\n========================================\n");
    printf(" Unified memory demo complete!\n");
    printf("========================================\n");

    return 0;
}
