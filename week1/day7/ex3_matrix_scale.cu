#include <stdio.h>

/**
 * 練習 3：矩陣縮放
 * 將矩陣中每個元素乘以常數 k
 */

__global__ void matrixScale(float *matrix, float k, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
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
    printf("    練習 3：矩陣縮放\n");
    printf("========================================\n\n");

    const int width = 4;
    const int height = 3;
    const float k = 2.5f;
    size_t bytes = width * height * sizeof(float);

    // 使用統一記憶體
    float *matrix;
    cudaMallocManaged(&matrix, bytes);

    // 初始化矩陣
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            matrix[y * width + x] = (float)(y * width + x + 1);
        }
    }

    printMatrix(matrix, width, height, "原始矩陣");
    printf("縮放係數 k = %.1f\n\n", k);

    // 執行核心函數
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + 15) / 16, (height + 15) / 16);

    matrixScale<<<numBlocks, threadsPerBlock>>>(matrix, k, width, height);
    cudaDeviceSynchronize();

    printMatrix(matrix, width, height, "縮放後的矩陣 (×2.5)");

    // 驗證
    bool correct = true;
    for (int i = 0; i < width * height; i++) {
        float expected = (i + 1) * k;
        if (abs(matrix[i] - expected) > 0.001f) {
            correct = false;
            break;
        }
    }
    printf("結果驗證: %s\n", correct ? " 正確" : " 錯誤");

    // 釋放記憶體
    cudaFree(matrix);

    return 0;
}
