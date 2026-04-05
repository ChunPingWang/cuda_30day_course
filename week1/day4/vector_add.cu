#include <stdio.h>
#include <stdlib.h>

/**
 * CUDA kernel function: Vector Addition
 * Each thread computes one element
 */
// __global__ 表示這是一個「核心函式」(kernel)，由 CPU 呼叫、在 GPU 上執行
__global__ void vectorAdd(int *a, int *b, int *c, int n) {
    // 計算此執行緒的全域索引：哪一個 block × 每個 block 的執行緒數 + block 內的執行緒編號
    // blockIdx.x = 目前所在的 block 編號
    // blockDim.x = 每個 block 有幾個執行緒
    // threadIdx.x = 在 block 內的執行緒編號
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // ⚠️ 注意：一定要做邊界檢查！因為總執行緒數可能超過陣列大小 n
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    printf("========================================\n");
    printf("    CUDA Vector Addition Demo\n");
    printf("========================================\n\n");

    // Set array size
    const int n = 10;
    size_t bytes = n * sizeof(int);

    // ========== Step 1: Allocate host memory and initialize ==========
    printf("Step 1: Prepare data on host\n");

    int *h_A = (int*)malloc(bytes);  // Host array A
    int *h_B = (int*)malloc(bytes);  // Host array B
    int *h_C = (int*)malloc(bytes);  // Host array C (for results)

    // Initialize data
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

    // ========== Step 2: Allocate device memory ==========
    printf("Step 2: Allocate memory on GPU\n");

    int *d_A, *d_B, *d_C;  // Device（GPU）指標，指向 GPU 上的記憶體
    // cudaMalloc 在 GPU 上配置記憶體（類似 CPU 的 malloc）
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    printf("Allocated %zu bytes x 3 = %zu bytes on GPU\n\n", bytes, bytes * 3);

    // ========== Step 3: Copy data from host to device ==========
    printf("Step 3: Copy data from CPU to GPU\n");
    // cudaMemcpy 把資料從 CPU（Host）複製到 GPU（Device）
    // 💡 Debug 提示：如果結果全是 0，可能是忘了這一步
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    printf("Data transfer complete!\n\n");

    // ========== Step 4: Launch kernel function ==========
    printf("Step 4: Launch GPU kernel function\n");

    int threadsPerBlock = 4;  // 每個 block 有 4 個執行緒
    // 無條件進位除法，確保有足夠的 block 覆蓋所有元素
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    printf("Config: <<<{%d}, {%d}>>>\n", blocks, threadsPerBlock);
    printf("Total threads: %d\n\n", blocks * threadsPerBlock);

    // <<<blocks, threadsPerBlock>>> 是 CUDA 特有的啟動語法，指定 grid 和 block 的大小
    vectorAdd<<<blocks, threadsPerBlock>>>(d_A, d_B, d_C, n);

    // cudaDeviceSynchronize() 讓 CPU 等待 GPU 完成所有工作
    // ⚠️ 注意：如果沒有這行，CPU 可能在 GPU 還沒算完就去讀結果
    cudaDeviceSynchronize();

    // ========== Step 5: Copy results from device to host ==========
    printf("Step 5: Copy results from GPU to CPU\n");
    // 把結果從 GPU（Device）複製回 CPU（Host）
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    printf("Results transfer complete!\n\n");

    // ========== Step 6: Display results ==========
    printf("Step 6: Display results\n");
    printf("C = A + B = [ ");
    for (int i = 0; i < n; i++) {
        printf("%d ", h_C[i]);
    }
    printf("]\n\n");

    // Verify results
    printf("Verifying results...\n");
    bool correct = true;
    for (int i = 0; i < n; i++) {
        if (h_C[i] != h_A[i] + h_B[i]) {
            printf(" Error: C[%d] = %d, expected %d\n", i, h_C[i], h_A[i] + h_B[i]);
            correct = false;
        }
    }
    if (correct) {
        printf(" All results correct!\n\n");
    }

    // ========== Step 7: Free memory ==========
    printf("Step 7: Free memory\n");
    // cudaFree 釋放 GPU 記憶體（對應 cudaMalloc）
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    // free 釋放 CPU 記憶體（對應 malloc）
    // ⚠️ 注意：GPU 和 CPU 的記憶體要分別釋放，不能搞混
    free(h_A);
    free(h_B);
    free(h_C);
    printf("Memory freed\n");

    printf("\n========================================\n");
    printf(" Program execution complete!\n");
    printf("========================================\n");

    return 0;
}
