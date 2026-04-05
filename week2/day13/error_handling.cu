#include <stdio.h>
#include <stdlib.h>

/**
 * Day 13: CUDA 錯誤處理範例
 *
 * 展示正確的錯誤處理方式
 */

// 錯誤檢查巨集：包裝任何 CUDA API 呼叫，自動檢查回傳值
// 💡 Debug 提示：每個 cudaMalloc / cudaMemcpy / cudaFree 都應該用 CHECK_CUDA 包裝
// __FILE__ 和 __LINE__ 是編譯器巨集，會自動替換成檔名和行號，方便定位錯誤
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s (error code: %d)\n", \
                cudaGetErrorString(err), err); \
        fprintf(stderr, "    at %s:%d\n", __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// 核心函數錯誤檢查：放在 <<<>>> 啟動之後
// ⚠️ 注意：核心函式啟動是非同步的，不會回傳錯誤碼，必須用 cudaGetLastError() 檢查
#define CHECK_KERNEL() do { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "Kernel Launch Error: %s\n", cudaGetErrorString(err)); \
        fprintf(stderr, "    at %s:%d\n", __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// 正確的核心函數：有邊界檢查
__global__ void safeKernel(int *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {  // ⚠️ 注意：邊界檢查是防止越界存取的關鍵，絕對不能省略
        data[idx] = idx * 2;
    }
}

// 會觸發錯誤的核心函數（示範用）
__global__ void unsafeKernel(int *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // ⚠️ 注意：沒有邊界檢查！當 idx >= n 時會存取到未分配的記憶體
    // 💡 Debug 提示：可用 compute-sanitizer 工具偵測此類越界錯誤
    data[idx] = idx * 2;
}

void demonstrateErrorHandling() {
    printf("=== 錯誤處理示範 ===\n\n");

    int n = 1000;
    int *d_data;
    size_t size = n * sizeof(int);

    // 1. 正確使用錯誤檢查：CHECK_CUDA 會自動處理錯誤
    printf("1. 分配 GPU 記憶體...\n");
    CHECK_CUDA(cudaMalloc(&d_data, size));  // 如果分配失敗，會印出錯誤訊息並結束程式
    printf("   成功！\n\n");

    // 2. 執行核心函數
    printf("2. 執行核心函數...\n");
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    safeKernel<<<blocks, threads>>>(d_data, n);  // <<<blocks, threads>>> 啟動核心函式
    CHECK_KERNEL();  // 檢查核心啟動是否成功（例如 threads 是否超過上限）
    CHECK_CUDA(cudaDeviceSynchronize());  // 等待 GPU 執行完成，同時檢查執行期間的錯誤
    printf("   成功！\n\n");

    // 3. 複製結果
    printf("3. 複製結果到主機...\n");
    int *h_data = (int*)malloc(size);
    CHECK_CUDA(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));
    printf("   成功！\n");
    printf("   前 5 個結果: %d, %d, %d, %d, %d\n\n",
           h_data[0], h_data[1], h_data[2], h_data[3], h_data[4]);

    // 4. 清理
    printf("4. 釋放記憶體...\n");
    CHECK_CUDA(cudaFree(d_data));
    free(h_data);
    printf("   成功！\n\n");
}

void demonstrateCommonErrors() {
    printf("=== 常見錯誤類型 ===\n\n");

    // 錯誤 1：分配過大的記憶體
    printf("1. 嘗試分配過大的記憶體...\n");
    int *huge_ptr;
    cudaError_t err = cudaMalloc(&huge_ptr, (size_t)1024 * 1024 * 1024 * 100);  // 100 GB，超過 GPU 記憶體
    if (err != cudaSuccess) {
        printf("   預期的錯誤: %s\n\n", cudaGetErrorString(err));
        // 💡 Debug 提示：CUDA 錯誤會「黏住」，必須用 cudaGetLastError() 清除，否則後續 API 都會失敗
        cudaGetLastError();  // 清除錯誤狀態
    }

    // 錯誤 2：無效的核心配置
    printf("2. 嘗試使用無效的核心配置...\n");
    int *d_data;
    cudaMalloc(&d_data, 100 * sizeof(int));

    // ⚠️ 注意：每個 Block 最多 1024 個執行緒（硬體限制）
    safeKernel<<<1, 2048>>>(d_data, 100);  // 2048 > 1024，啟動會失敗
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("   預期的錯誤: %s\n\n", cudaGetErrorString(err));
    }

    cudaFree(d_data);

    // 錯誤 3：無效的記憶體複製方向
    printf("3. 無效的記憶體操作（已被攔截）...\n");
    int *h_data = (int*)malloc(100 * sizeof(int));
    int *d_data2;
    cudaMalloc(&d_data2, 100 * sizeof(int));

    // 這會成功，但使用錯誤的指標會失敗
    // cudaMemcpy(h_data, h_data, 100 * sizeof(int), cudaMemcpyDeviceToHost);

    printf("   跳過危險操作\n\n");

    free(h_data);
    cudaFree(d_data2);
}

// 查詢並印出 GPU 硬體資訊（了解硬體限制有助於 debug）
void printDeviceInfo() {
    printf("=== GPU 資訊 ===\n\n");

    int deviceCount;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));  // 取得系統中 CUDA 裝置的數量

    if (deviceCount == 0) {
        printf("找不到 CUDA 設備！\n");
        return;
    }

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;  // cudaDeviceProp 結構體包含 GPU 所有硬體規格
        CHECK_CUDA(cudaGetDeviceProperties(&prop, i));

        printf("設備 %d: %s\n", i, prop.name);
        printf("  計算能力: %d.%d\n", prop.major, prop.minor);
        printf("  全域記憶體: %.1f GB\n", prop.totalGlobalMem / 1e9);
        printf("  每個 Block 最大執行緒數: %d\n", prop.maxThreadsPerBlock);
        printf("  每個 SM 最大執行緒數: %d\n", prop.maxThreadsPerMultiProcessor);
        printf("\n");
    }
}

int main() {
    printf("========================================\n");
    printf("    CUDA 錯誤處理與除錯技巧\n");
    printf("========================================\n\n");

    printDeviceInfo();
    demonstrateErrorHandling();
    demonstrateCommonErrors();

    printf("========================================\n");
    printf("    除錯建議\n");
    printf("========================================\n");
    printf("1. 始終使用錯誤檢查巨集\n");
    printf("2. 在核心函數後檢查 cudaGetLastError()\n");
    printf("3. 使用 compute-sanitizer 檢測記憶體錯誤\n");
    printf("4. 在核心中使用 printf 進行除錯\n");
    printf("5. 邊界檢查是必要的！\n");

    return 0;
}
