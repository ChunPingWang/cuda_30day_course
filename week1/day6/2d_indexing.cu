#include <stdio.h>

/**
 * 示範 2D 索引的 kernel：印出每個執行緒對應的 (x, y) 座標和 1D 索引
 */
__global__ void show2DIndex(int width, int height) {
    // 2D 索引：x 方向和 y 方向各自計算
    // blockIdx.x/y = 目前 block 在 grid 中的 x/y 座標
    // blockDim.x/y = 每個 block 在 x/y 方向的執行緒數量
    // threadIdx.x/y = 在 block 內的 x/y 編號
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        // 💡 Debug 提示：2D -> 1D 的轉換公式：idx = y * width + x（Row-Major 排列）
        int idx = y * width + x;
        printf("(x=%d, y=%d) -> 1D index = %d\n", x, y, idx);
    }
}

/**
 * 用 2D 索引填入矩陣：每個元素 = x + y * 10
 */
__global__ void fill2DMatrix(int *matrix, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // 行（column）索引
    int y = blockIdx.y * blockDim.y + threadIdx.y;  // 列（row）索引

    // ⚠️ 注意：2D 的邊界檢查要同時檢查 x 和 y
    if (x < width && y < height) {
        int idx = y * width + x;  // 2D 座標轉成 1D 索引
        matrix[idx] = x + y * 10;
    }
}

/**
 * 矩陣轉置：將 input(y, x) 的值放到 output(x, y)
 */
__global__ void transpose(int *input, int *output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int inputIdx = y * width + x;      // 原矩陣的 (y, x) 位置
        int outputIdx = x * height + y;    // 轉置後的 (x, y) 位置
        // ⚠️ 注意：轉置後矩陣的寬高互換，所以 outputIdx 用 height 而不是 width
        output[outputIdx] = input[inputIdx];
    }
}

/**
 * Print matrix
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
    printf("    2D Indexing and Matrix Operations\n");
    printf("========================================\n\n");

    // ========== Demo 1: 2D Indexing ==========
    printf("Demo 1: 2D Index Mapping\n");
    printf("Matrix size: 4 x 3\n");
    printf("Block size: 2 x 2\n\n");

    // dim3 是 CUDA 的三維向量型別，用來設定 2D/3D 的 block 和 grid 大小
    dim3 threadsPerBlock(2, 2);   // 每個 block 有 2x2 = 4 個執行緒
    dim3 numBlocks(2, 2);  // grid 有 2x2 = 4 個 block（4/2, ceil(3/2)）

    show2DIndex<<<numBlocks, threadsPerBlock>>>(4, 3);
    cudaDeviceSynchronize();

    // ========== Demo 2: 2D Matrix Fill ==========
    printf("\nDemo 2: Fill matrix using 2D indexing\n");

    const int width = 5;
    const int height = 4;
    size_t bytes = width * height * sizeof(int);

    int *d_matrix;
    cudaMallocManaged(&d_matrix, bytes);  // 統一記憶體，CPU 和 GPU 共用

    dim3 threads(4, 4);  // 每個 block 4x4 = 16 個執行緒
    dim3 blocks((width + 3) / 4, (height + 3) / 4);  // 無條件進位，確保覆蓋整個矩陣

    fill2DMatrix<<<blocks, threads>>>(d_matrix, width, height);
    cudaDeviceSynchronize();

    printMatrix(d_matrix, width, height, "Filled matrix (value = x + y*10)");

    // ========== Demo 3: Matrix Transpose ==========
    printf("Demo 3: Matrix Transpose\n");

    int *d_transposed;
    cudaMallocManaged(&d_transposed, bytes);

    transpose<<<blocks, threads>>>(d_matrix, d_transposed, width, height);
    cudaDeviceSynchronize();

    printMatrix(d_transposed, height, width, "Transposed matrix");

    // Verify transpose result
    printf("Verify transpose result:\n");
    printf("  Original (1,2) = %d\n", d_matrix[2 * width + 1]);
    printf("  Transposed (2,1) = %d\n", d_transposed[1 * height + 2]);
    printf("  Should be equal: %s\n\n",
           d_matrix[2 * width + 1] == d_transposed[1 * height + 2] ? "YES" : "NO");

    // ========== Cleanup ==========
    cudaFree(d_matrix);
    cudaFree(d_transposed);

    printf("========================================\n");
    printf("2D Index Formula:\n");
    printf("   idx = y * width + x\n");
    printf("   (Row-Major Order)\n");
    printf("========================================\n");

    return 0;
}
