#include <stdio.h>

/**
 * 展示執行緒索引的計算
 */
__global__ void printThreadInfo() {
    // 計算全域索引
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

    // 每個執行緒印出自己的資訊
    printf("Block: %2d | Thread: %2d | Global Index: %2d\n",
           blockIdx.x, threadIdx.x, globalIdx);
}

/**
 * 展示二維索引
 */
__global__ void print2DThreadInfo() {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    printf("Block(%d,%d) Thread(%d,%d) => Global(%d,%d)\n",
           blockIdx.x, blockIdx.y,
           threadIdx.x, threadIdx.y,
           x, y);
}

int main() {
    printf("========================================\n");
    printf("      CUDA 執行緒索引示範\n");
    printf("========================================\n\n");

    // 示範 1: 一維索引
    printf("示範 1: 一維索引\n");
    printf("配置: <<<3, 4>>> (3 個 Block，每個 4 個 Thread)\n");
    printf("----------------------------------------\n");
    printThreadInfo<<<3, 4>>>();
    cudaDeviceSynchronize();

    printf("\n示範 2: 不同的配置\n");
    printf("配置: <<<2, 8>>> (2 個 Block，每個 8 個 Thread)\n");
    printf("----------------------------------------\n");
    printThreadInfo<<<2, 8>>>();
    cudaDeviceSynchronize();

    // 示範 3: 二維索引
    printf("\n示範 3: 二維索引\n");
    printf("配置: <<<(2,2), (2,2)>>> (2x2 Block Grid, 2x2 Thread Block)\n");
    printf("----------------------------------------\n");
    dim3 blocks(2, 2);
    dim3 threads(2, 2);
    print2DThreadInfo<<<blocks, threads>>>();
    cudaDeviceSynchronize();

    printf("\n========================================\n");
    printf("💡 重要觀察：\n");
    printf("1. 全域索引 = blockIdx.x * blockDim.x + threadIdx.x\n");
    printf("2. 執行緒的執行順序是不確定的！\n");
    printf("3. 不同配置可以產生相同數量的執行緒\n");
    printf("========================================\n");

    return 0;
}
