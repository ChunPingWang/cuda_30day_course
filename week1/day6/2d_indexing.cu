#include <stdio.h>

/**
 * Kernel to demonstrate 2D indexing
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
 * 2D matrix fill: each element = x + y * 10
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
 * Matrix transpose
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

    dim3 threadsPerBlock(2, 2);
    dim3 numBlocks(2, 2);  // (4/2, ceil(3/2))

    show2DIndex<<<numBlocks, threadsPerBlock>>>(4, 3);
    cudaDeviceSynchronize();

    // ========== Demo 2: 2D Matrix Fill ==========
    printf("\nDemo 2: Fill matrix using 2D indexing\n");

    const int width = 5;
    const int height = 4;
    size_t bytes = width * height * sizeof(int);

    int *d_matrix;
    cudaMallocManaged(&d_matrix, bytes);

    dim3 threads(4, 4);
    dim3 blocks((width + 3) / 4, (height + 3) / 4);

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
