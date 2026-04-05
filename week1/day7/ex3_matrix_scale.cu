#include <stdio.h>
#include <math.h>

/**
 * 練習 3：矩陣縮放
 * 將矩陣中每個元素乘以常數 k
 */

// 使用 2D 索引的 kernel：每個執行緒處理矩陣中的一個元素
__global__ void matrixScale(float *matrix, float k, int width, int height) {
    // 2D 執行緒索引
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // 行（column）
    int y = blockIdx.y * blockDim.y + threadIdx.y;  // 列（row）

    // ⚠️ 注意：2D 邊界檢查，x 和 y 都要檢查
    if (x < width && y < height) {
        int idx = y * width + x;  // 2D -> 1D 索引轉換（Row-Major）
        matrix[idx] = matrix[idx] * k;
    }
}

void printMatrix(float *matrix, int width, int height, const char *name) {
    printf("%s:\n", name);
    for (int y = 0; y < height; y++) {
        printf("  [ ");
        for (int x = 0; x < width; x++) {
            printf("%5.1f ", matrix[y * width + x]);
        }
        printf("]\n");
    }
    printf("\n");
}

int main() {
    printf("========================================\n");
    printf("    Exercise 3: Matrix Scaling\n");
    printf("========================================\n\n");

    const int width = 4;
    const int height = 3;
    const float k = 2.5f;
    size_t bytes = width * height * sizeof(float);

    // 使用統一記憶體
    float *matrix;
    cudaMallocManaged(&matrix, bytes);

    // Initialize matrix
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            matrix[y * width + x] = (float)(y * width + x + 1);
        }
    }

    printMatrix(matrix, width, height, "Original Matrix");
    printf("Scale factor k = %.1f\n\n", k);

    // 啟動 kernel
    dim3 threadsPerBlock(16, 16);  // 每個 block 16x16 = 256 個執行緒
    dim3 numBlocks((width + 15) / 16, (height + 15) / 16);  // 無條件進位計算 grid 大小

    // <<<numBlocks, threadsPerBlock>>> 使用 dim3 設定 2D 的 grid 和 block
    matrixScale<<<numBlocks, threadsPerBlock>>>(matrix, k, width, height);
    cudaDeviceSynchronize();  // 等待 GPU 完成

    printMatrix(matrix, width, height, "Scaled Matrix (x2.5)");

    // Verify
    bool correct = true;
    for (int i = 0; i < width * height; i++) {
        float expected = (i + 1) * k;
        // 💡 Debug 提示：浮點數比較用容差，不能直接 ==
        if (fabs(matrix[i] - expected) > 0.001f) {
            correct = false;
            break;
        }
    }
    printf("Result verification: %s\n", correct ? "CORRECT" : "ERROR");

    // 釋放統一記憶體
    cudaFree(matrix);

    return 0;
}
