#include <stdio.h>

/**
 * 使用執行緒索引來存取陣列元素
 * 每個執行緒處理一個陣列元素
 */
__global__ void fillArray(int *arr, int n) {
    // 計算全域索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 確保不超出陣列範圍
    if (idx < n) {
        // 每個執行緒將自己的索引值存入陣列
        arr[idx] = idx * 10;
    }
}

/**
 * 展示邊界檢查的重要性
 */
__global__ void printIndexInfo(int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        printf(" Thread %d: 處理 array[%d]\n", idx, idx);
    } else {
        printf(" Thread %d: 超出範圍，閒置\n", idx);
    }
}

int main() {
    printf("========================================\n");
    printf("   使用執行緒索引處理陣列\n");
    printf("========================================\n\n");

    const int n = 10;  // 陣列大小
    int *d_arr;        // 設備端陣列指標

    // 在 GPU 上分配記憶體
    cudaMalloc(&d_arr, n * sizeof(int));

    // 核心函數配置
    int threadsPerBlock = 4;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    printf("陣列大小: %d\n", n);
    printf("每個 Block 的執行緒數: %d\n", threadsPerBlock);
    printf("Block 數量: %d\n", blocks);
    printf("總執行緒數: %d\n\n", blocks * threadsPerBlock);

    // 示範 1: 展示哪些執行緒會被使用
    printf("示範 1: 執行緒使用情況\n");
    printf("----------------------------------------\n");
    printIndexInfo<<<blocks, threadsPerBlock>>>(n);
    cudaDeviceSynchronize();

    // 示範 2: 填充陣列
    printf("\n示範 2: 使用 GPU 填充陣列\n");
    printf("----------------------------------------\n");
    fillArray<<<blocks, threadsPerBlock>>>(d_arr, n);
    cudaDeviceSynchronize();

    // 將結果複製回主機並顯示
    int h_arr[n];
    cudaMemcpy(h_arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);

    printf("陣列內容: [ ");
    for (int i = 0; i < n; i++) {
        printf("%d ", h_arr[i]);
    }
    printf("]\n");

    // 釋放記憶體
    cudaFree(d_arr);

    printf("\n========================================\n");
    printf("💡 關鍵概念：\n");
    printf("1. idx = blockIdx.x * blockDim.x + threadIdx.x\n");
    printf("2. 必須檢查 idx < n 避免越界存取\n");
    printf("3. 執行緒數量可能大於資料數量\n");
    printf("\n🎯 練習：\n");
    printf("嘗試修改 threadsPerBlock 的值（如 8, 16）\n");
    printf("觀察需要多少個 Block 和閒置的執行緒數量\n");
    printf("========================================\n");

    return 0;
}
