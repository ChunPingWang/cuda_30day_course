#include <stdio.h>
#include <stdlib.h>

/**
 * CUDA 核心函數：向量加法
 * 每個執行緒負責計算一個元素
 */
__global__ void vectorAdd(int *a, int *b, int *c, int n) {
    // 計算全域索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 邊界檢查：確保不超出陣列範圍
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    printf("========================================\n");
    printf("    CUDA 向量加法示範\n");
    printf("========================================\n\n");

    // 設定陣列大小
    const int n = 10;
    size_t bytes = n * sizeof(int);

    // ========== 步驟 1：在主機端分配記憶體並初始化 ==========
    printf("步驟 1: 在主機端準備資料\n");

    int *h_A = (int*)malloc(bytes);  // 主機端陣列 A
    int *h_B = (int*)malloc(bytes);  // 主機端陣列 B
    int *h_C = (int*)malloc(bytes);  // 主機端陣列 C（存結果）

    // 初始化資料
    printf("A = [ ");
    for (int i = 0; i < n; i++) {
        h_A[i] = i;
        printf("%d ", h_A[i]);
    }
    printf("]\n");

    printf("B = [ ");
    for (int i = 0; i < n; i++) {
        h_B[i] = i * 2;
        printf("%d ", h_B[i]);
    }
    printf("]\n\n");

    // ========== 步驟 2：在設備端分配記憶體 ==========
    printf("步驟 2: 在 GPU 上分配記憶體\n");

    int *d_A, *d_B, *d_C;  // 設備端指標
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    printf("已在 GPU 上分配 %zu bytes × 3 = %zu bytes\n\n", bytes, bytes * 3);

    // ========== 步驟 3：將資料從主機複製到設備 ==========
    printf("步驟 3: 將資料從 CPU 複製到 GPU\n");
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    printf("資料傳輸完成！\n\n");

    // ========== 步驟 4：啟動核心函數 ==========
    printf("步驟 4: 啟動 GPU 核心函數\n");

    int threadsPerBlock = 4;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    printf("配置: <<<{%d}, {%d}>>>\n", blocks, threadsPerBlock);
    printf("總執行緒數: %d\n\n", blocks * threadsPerBlock);

    vectorAdd<<<blocks, threadsPerBlock>>>(d_A, d_B, d_C, n);

    // 等待 GPU 完成
    cudaDeviceSynchronize();

    // ========== 步驟 5：將結果從設備複製回主機 ==========
    printf("步驟 5: 將結果從 GPU 複製回 CPU\n");
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    printf("結果傳輸完成！\n\n");

    // ========== 步驟 6：顯示結果 ==========
    printf("步驟 6: 顯示結果\n");
    printf("C = A + B = [ ");
    for (int i = 0; i < n; i++) {
        printf("%d ", h_C[i]);
    }
    printf("]\n\n");

    // 驗證結果
    printf("驗證結果...\n");
    bool correct = true;
    for (int i = 0; i < n; i++) {
        if (h_C[i] != h_A[i] + h_B[i]) {
            printf(" 錯誤：C[%d] = %d, 預期 %d\n", i, h_C[i], h_A[i] + h_B[i]);
            correct = false;
        }
    }
    if (correct) {
        printf(" 所有結果正確！\n\n");
    }

    // ========== 步驟 7：釋放記憶體 ==========
    printf("步驟 7: 釋放記憶體\n");
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    printf("記憶體已釋放\n");

    printf("\n========================================\n");
    printf(" 程式執行完成！\n");
    printf("========================================\n");

    return 0;
}
