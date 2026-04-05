#include <stdio.h>

/**
 * 示範如何計算執行緒的全域索引（Global Index）
 * 每個執行緒都有自己獨一無二的全域索引
 */
__global__ void printThreadInfo() {
    // 計算全域索引的公式：blockIdx.x * blockDim.x + threadIdx.x
    // blockIdx.x  = 這個執行緒所在的 Block 編號
    // blockDim.x  = 每個 Block 裡有幾個 Thread
    // threadIdx.x = 這個執行緒在 Block 內的編號
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread prints its own info
    printf("Block: %2d | Thread: %2d | Global Index: %2d\n",
           blockIdx.x, threadIdx.x, globalIdx);
}

/**
 * 示範 2D（二維）索引計算
 * 適合用在矩陣、圖片等二維資料結構
 */
__global__ void print2DThreadInfo() {
    int x = blockIdx.x * blockDim.x + threadIdx.x; // 水平方向的全域索引
    int y = blockIdx.y * blockDim.y + threadIdx.y;  // 垂直方向的全域索引

    printf("Block(%d,%d) Thread(%d,%d) => Global(%d,%d)\n",
           blockIdx.x, blockIdx.y,
           threadIdx.x, threadIdx.y,
           x, y);
}

int main() {
    printf("========================================\n");
    printf("      CUDA Thread Index Demo\n");
    printf("========================================\n\n");

    // 示範 1：一維索引
    printf("Demo 1: 1D Indexing\n");
    printf("Config: <<<3, 4>>> (3 Blocks, 4 Threads each)\n");
    printf("----------------------------------------\n");
    printThreadInfo<<<3, 4>>>(); // 啟動 3 個 Block，每個有 4 個 Thread，共 12 個執行緒
    cudaDeviceSynchronize();

    printf("\nDemo 2: Different configuration\n");
    printf("Config: <<<2, 8>>> (2 Blocks, 8 Threads each)\n");
    printf("----------------------------------------\n");
    printThreadInfo<<<2, 8>>>();
    cudaDeviceSynchronize();

    // 示範 3：二維索引
    printf("\nDemo 3: 2D Indexing\n");
    printf("Config: <<<(2,2), (2,2)>>> (2x2 Block Grid, 2x2 Thread Block)\n");
    printf("----------------------------------------\n");
    dim3 blocks(2, 2);   // dim3 是 CUDA 的三維向量型別，這裡建立 2x2 的 Grid
    dim3 threads(2, 2);  // 每個 Block 內有 2x2 = 4 個 Thread
    print2DThreadInfo<<<blocks, threads>>>(); // 總共 2x2 x 2x2 = 16 個執行緒
    cudaDeviceSynchronize();

    printf("\n========================================\n");
    printf("Key Observations:\n");
    printf("1. Global Index = blockIdx.x * blockDim.x + threadIdx.x\n");
    printf("2. Thread execution order is non-deterministic!\n");
    printf("3. Different configs can create same number of threads\n");
    printf("========================================\n");

    return 0;
}
