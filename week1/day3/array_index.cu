#include <stdio.h>

/**
 * 用執行緒索引來存取陣列元素
 * 每個執行緒負責處理一個陣列元素（一對一映射）
 */
__global__ void fillArray(int *arr, int n) {
    // 計算這個執行緒對應的全域索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // ⚠️ 注意：一定要做邊界檢查！因為執行緒總數可能超過陣列大小
    // 💡 Debug 提示：如果忘了這個 if 判斷，會導致記憶體越界存取（undefined behavior）
    if (idx < n) {
        arr[idx] = idx * 10; // 每個執行緒把自己的索引值 * 10 存入陣列
    }
}

/**
 * 示範邊界檢查的重要性
 * 超出陣列範圍的執行緒必須被跳過，不能做任何資料操作
 */
__global__ void printIndexInfo(int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        printf(" Thread %d: Processing array[%d]\n", idx, idx); // 在有效範圍內，可以工作
    } else {
        printf(" Thread %d: Out of range, idle\n", idx); // 超出範圍，閒置不做事
    }
}

int main() {
    printf("========================================\n");
    printf("   Using Thread Index to Process Arrays\n");
    printf("========================================\n\n");

    const int n = 10;  // 陣列大小
    int *d_arr;        // 指向 GPU（Device）記憶體的指標，前綴 d_ 代表 device

    // 在 GPU 上配置記憶體（類似 CPU 的 malloc，但配置在 GPU 的 VRAM 上）
    // ⚠️ 注意：d_arr 指向 GPU 記憶體，不能在 CPU 端直接讀寫！
    cudaMalloc(&d_arr, n * sizeof(int));

    // Kernel 啟動設定
    int threadsPerBlock = 4;
    // 💡 Debug 提示：這個公式是「無條件進位除法」，確保 Block 數量足夠涵蓋所有元素
    // 例如 n=10, threadsPerBlock=4 → blocks = (10+4-1)/4 = 13/4 = 3（共 12 個執行緒 >= 10）
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    printf("Array size: %d\n", n);
    printf("Threads per Block: %d\n", threadsPerBlock);
    printf("Number of Blocks: %d\n", blocks);
    printf("Total threads: %d\n\n", blocks * threadsPerBlock);

    // 示範 1：顯示哪些執行緒有在工作、哪些閒置
    printf("Demo 1: Thread usage status\n");
    printf("----------------------------------------\n");
    printIndexInfo<<<blocks, threadsPerBlock>>>(n); // <<<Block數, 每Block的Thread數>>>
    cudaDeviceSynchronize(); // 等 GPU 印完再繼續

    // 示範 2：用 GPU 填充陣列
    printf("\nDemo 2: Fill array using GPU\n");
    printf("----------------------------------------\n");
    fillArray<<<blocks, threadsPerBlock>>>(d_arr, n);
    cudaDeviceSynchronize();

    // 將結果從 GPU 記憶體複製回 CPU 記憶體，才能在 CPU 端印出
    // cudaMemcpyDeviceToHost = 從 Device（GPU）複製到 Host（CPU）
    int h_arr[10]; // 前綴 h_ 代表 host（CPU 端）
    cudaMemcpy(h_arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Array contents: [ ");
    for (int i = 0; i < n; i++) {
        printf("%d ", h_arr[i]);
    }
    printf("]\n");

    // 釋放 GPU 記憶體（類似 CPU 的 free）
    // ⚠️ 注意：忘記 cudaFree 會造成 GPU 記憶體洩漏
    cudaFree(d_arr);

    printf("\n========================================\n");
    printf("Key Concepts:\n");
    printf("1. idx = blockIdx.x * blockDim.x + threadIdx.x\n");
    printf("2. Must check idx < n to avoid out-of-bounds access\n");
    printf("3. Thread count may exceed data count\n");
    printf("\nExercise:\n");
    printf("Try modifying threadsPerBlock value (e.g., 8, 16)\n");
    printf("Observe how many Blocks and idle threads are needed\n");
    printf("========================================\n");

    return 0;
}
