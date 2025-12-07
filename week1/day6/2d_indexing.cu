#include <stdio.h>

/**
 * 展示 2D 索引的核心函數
 */
__global__ void show2DIndex(int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        printf("(x=%d, y=%d) -> 1D index = %d\n", x, y, idx);
    }
}

/**
 * 2D 矩陣填充：每個元素 = x + y * 10
 */
__global__ void fill2DMatrix(int *matrix, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        matrix[idx] = x + y * 10;
    }
}

/**
 * 矩陣轉置
 */
__global__ void transpose(int *input, int *output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int inputIdx = y * width + x;
        int outputIdx = x * height + y;
        output[outputIdx] = input[inputIdx];
    }
}

/**
 * 印出矩陣
 */
void printMatrix(int *matrix, int width, int height, const char *name) {
    printf("%s (%d x %d):\n", name, width, height);
    for (int y = 0; y < height; y++) {
        printf("  ");
        for (int x = 0; x < width; x++) {
            printf("%3d ", matrix[y * width + x]);
        }
        printf("\n");
    }
    printf("\n");
}

int main() {
    printf("========================================\n");
    printf("    2D 索引與矩陣操作示範\n");
    printf("========================================\n\n");

    // ========== 示範 1：2D 索引 ==========
    printf("示範 1: 2D 索引映射\n");
    printf("矩陣大小: 4 x 3\n");
    printf("Block 大小: 2 x 2\n\n");

    dim3 threadsPerBlock(2, 2);
    dim3 numBlocks(2, 2);  // (4/2, 3/2 向上取整)

    show2DIndex<<<numBlocks, threadsPerBlock>>>(4, 3);
    cudaDeviceSynchronize();

    // ========== 示範 2：2D 矩陣填充 ==========
    printf("\n示範 2: 使用 2D 索引填充矩陣\n");

    const int width = 5;
    const int height = 4;
    size_t bytes = width * height * sizeof(int);

    int *d_matrix;
    cudaMallocManaged(&d_matrix, bytes);

    dim3 threads(4, 4);
    dim3 blocks((width + 3) / 4, (height + 3) / 4);

    fill2DMatrix<<<blocks, threads>>>(d_matrix, width, height);
    cudaDeviceSynchronize();

    printMatrix(d_matrix, width, height, "填充後的矩陣 (value = x + y*10)");

    // ========== 示範 3：矩陣轉置 ==========
    printf("示範 3: 矩陣轉置\n");

    int *d_transposed;
    cudaMallocManaged(&d_transposed, bytes);

    transpose<<<blocks, threads>>>(d_matrix, d_transposed, width, height);
    cudaDeviceSynchronize();

    printMatrix(d_transposed, height, width, "轉置後的矩陣");

    // 驗證轉置結果
    printf("驗證轉置結果：\n");
    printf("  原矩陣 (1,2) = %d\n", d_matrix[2 * width + 1]);
    printf("  轉置後 (2,1) = %d\n", d_transposed[1 * height + 2]);
    printf("  應該相等: %s\n\n",
           d_matrix[2 * width + 1] == d_transposed[1 * height + 2] ? "" : "");

    // ========== 清理 ==========
    cudaFree(d_matrix);
    cudaFree(d_transposed);

    printf("========================================\n");
    printf("💡 2D 索引公式：\n");
    printf("   idx = y * width + x\n");
    printf("   (行優先，Row-Major Order)\n");
    printf("========================================\n");

    return 0;
}
