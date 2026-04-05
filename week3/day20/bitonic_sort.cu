#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/**
 * Day 20: Bitonic Sort 雙調排序
 *
 * 展示 GPU 友好的排序演算法
 */

#define BLOCK_SIZE 256

// 全域記憶體版本：Bitonic Sort 的一個步驟
// 每次呼叫只做一輪「比較並交換」，需要外層迴圈多次呼叫
__global__ void bitonicSortStep(int *data, int j, int k, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    // XOR 運算找到配對的元素索引（Bitonic Sort 的核心配對邏輯）
    // 💡 Debug 提示：可以印出 idx 和 ixj 觀察配對關係
    int ixj = idx ^ j;

    // ixj > idx 確保每對只處理一次（避免重複交換）
    if (ixj > idx) {
        if ((idx & k) == 0) {
            // 這一段要排成升序
            if (data[idx] > data[ixj]) {
                int temp = data[idx];
                data[idx] = data[ixj];
                data[ixj] = temp;
            }
        } else {
            // 這一段要排成降序
            if (data[idx] < data[ixj]) {
                int temp = data[idx];
                data[idx] = data[ixj];
                data[ixj] = temp;
            }
        }
    }
}

// 共享記憶體版本（單一 Block 內排序 — 所有比較都在快速的共享記憶體中完成）
__global__ void bitonicSortShared(int *data, int n) {
    __shared__ int sdata[BLOCK_SIZE];  // 共享記憶體，block 內的執行緒共用

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // 載入資料到共享記憶體（超出範圍的填 INT_MAX，排序後會自動到最後面）
    sdata[tid] = (idx < n) ? data[idx] : INT_MAX;
    __syncthreads();

    // Bitonic Sort 主迴圈
    // k：控制「雙調序列」的長度（2→4→8→...→BLOCK_SIZE）
    for (int k = 2; k <= BLOCK_SIZE; k *= 2) {
        // j：控制比較的距離（從 k/2 遞減到 1）
        for (int j = k / 2; j > 0; j /= 2) {
            int ixj = tid ^ j;  // XOR 找配對

            if (ixj > tid && ixj < BLOCK_SIZE) {
                if ((tid & k) == 0) {
                    if (sdata[tid] > sdata[ixj]) {
                        int temp = sdata[tid];
                        sdata[tid] = sdata[ixj];
                        sdata[ixj] = temp;
                    }
                } else {
                    if (sdata[tid] < sdata[ixj]) {
                        int temp = sdata[tid];
                        sdata[tid] = sdata[ixj];
                        sdata[ixj] = temp;
                    }
                }
            }
            __syncthreads();  // ⚠️ 注意：每一步都必須同步，否則會讀到未更新的值
        }
    }

    // 將排序結果寫回全域記憶體
    if (idx < n) {
        data[idx] = sdata[tid];
    }
}

// CPU 排序（驗證用）
int compare(const void *a, const void *b) {
    return (*(int*)a - *(int*)b);
}

// 從 CPU 端驅動 Bitonic Sort：反覆啟動 kernel 完成所有步驟
// ⚠️ 注意：每個 kernel 呼叫只做一步比較交換，所以需要 O(log²n) 次 kernel 啟動
// 💡 Debug 提示：如果排序結果不正確，先用小陣列（如 n=8）印出每步的中間結果
void bitonicSortHost(int *d_data, int n) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // k：雙調序列長度（2,4,8,...,n）
    for (int k = 2; k <= n; k *= 2) {
        // j：比較距離（k/2, k/4, ..., 1）
        for (int j = k / 2; j > 0; j /= 2) {
            bitonicSortStep<<<blocks, BLOCK_SIZE>>>(d_data, j, k, n);
            cudaDeviceSynchronize();  // 等待 GPU 完成，確保下一步讀到正確的值
        }
    }
}

bool isSorted(int *arr, int n) {
    for (int i = 1; i < n; i++) {
        if (arr[i] < arr[i - 1]) {
            return false;
        }
    }
    return true;
}

bool arraysEqual(int *a, int *b, int n) {
    for (int i = 0; i < n; i++) {
        if (a[i] != b[i]) return false;
    }
    return true;
}

void printArray(const char *name, int *arr, int n, int limit) {
    printf("%s: [", name);
    for (int i = 0; i < n && i < limit; i++) {
        printf("%d", arr[i]);
        if (i < n - 1 && i < limit - 1) printf(", ");
    }
    if (n > limit) printf(", ...");
    printf("]\n");
}

int main() {
    printf("========================================\n");
    printf("    Bitonic Sort 雙調排序\n");
    printf("========================================\n\n");

    // 測試小陣列
    printf("【小陣列測試】\n");
    const int smallN = 16;
    int h_small[smallN];

    srand(42);
    for (int i = 0; i < smallN; i++) {
        h_small[i] = rand() % 100;
    }

    printArray("Before sort", h_small, smallN, 16);

    int *d_small;
    cudaMalloc(&d_small, smallN * sizeof(int));  // 在 GPU 分配記憶體
    cudaMemcpy(d_small, h_small, smallN * sizeof(int), cudaMemcpyHostToDevice);  // CPU → GPU

    bitonicSortHost(d_small, smallN);

    cudaMemcpy(h_small, d_small, smallN * sizeof(int), cudaMemcpyDeviceToHost);
    printArray("After sort", h_small, smallN, 16);
    printf("Verify: %s\n\n", isSorted(h_small, smallN) ? "PASS" : "FAIL");

    cudaFree(d_small);

    // 效能測試
    printf("【效能測試】\n");

    // 測試不同大小
    int sizes[] = {1024, 4096, 16384, 65536, 262144};
    int numSizes = sizeof(sizes) / sizeof(sizes[0]);

    printf("%-12s %-12s %-12s %-12s\n", "大小", "GPU時間(ms)", "CPU時間(ms)", "加速比");
    printf("------------------------------------------------\n");

    for (int s = 0; s < numSizes; s++) {
        int n = sizes[s];

        // ⚠️ 注意：Bitonic Sort 要求陣列大小必須是 2 的冪次，否則需要補齊
        int *h_data = (int*)malloc(n * sizeof(int));
        int *h_sorted = (int*)malloc(n * sizeof(int));

        for (int i = 0; i < n; i++) {
            h_data[i] = rand() % 10000;
            h_sorted[i] = h_data[i];
        }

        int *d_data;
        cudaMalloc(&d_data, n * sizeof(int));

        // GPU 計時
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaMemcpy(d_data, h_data, n * sizeof(int), cudaMemcpyHostToDevice);

        cudaEventRecord(start);
        bitonicSortHost(d_data, n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float gpuTime = 0;
        cudaEventElapsedTime(&gpuTime, start, stop);

        // CPU 計時
        clock_t cpuStart = clock();
        qsort(h_sorted, n, sizeof(int), compare);
        clock_t cpuEnd = clock();
        float cpuTime = (float)(cpuEnd - cpuStart) / CLOCKS_PER_SEC * 1000;

        // 驗證
        cudaMemcpy(h_data, d_data, n * sizeof(int), cudaMemcpyDeviceToHost);
        bool correct = arraysEqual(h_data, h_sorted, n);

        printf("%-12d %-12.3f %-12.3f %-12.2fx %s\n",
               n, gpuTime, cpuTime, cpuTime / gpuTime,
               correct ? "" : "(錯誤)");

        free(h_data);
        free(h_sorted);
        cudaFree(d_data);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    printf("\n");
    printf("========================================\n");
    printf("Bitonic Sort 特點：\n");
    printf("1. 比較網路是固定的，不依賴資料\n");
    printf("2. 時間複雜度: O(n log²n)\n");
    printf("3. 非常適合 GPU 平行化\n");
    printf("4. 需要 2 的冪次大小的陣列\n");
    printf("========================================\n");

    return 0;
}
