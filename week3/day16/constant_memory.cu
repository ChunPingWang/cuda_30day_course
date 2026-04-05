#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/**
 * Day 16: 常數記憶體範例
 *
 * 比較使用全域記憶體和常數記憶體的效能差異
 */

#define N 1000000
#define FILTER_SIZE 9

// __constant__ 宣告常數記憶體，存放在 GPU 的專用快取中
// ⚠️ 注意：常數記憶體最大只有 64KB，超過會編譯錯誤
// 特點：所有執行緒讀取同一位址時，只需一次記憶體存取（廣播機制）
__constant__ float c_filter[FILTER_SIZE];

// 使用全域記憶體的 1D 卷積（較慢的版本）
// 卷積：將濾波器滑過輸入陣列，計算加權總和
__global__ void convolutionGlobal(float *input, float *output,
                                   float *filter, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // 全域執行緒索引

    if (idx < n) {
        float sum = 0.0f;
        int halfSize = FILTER_SIZE / 2;  // 濾波器的半徑

        for (int i = 0; i < FILTER_SIZE; i++) {
            int inputIdx = idx - halfSize + i;
            // 邊界檢查：避免存取超出陣列範圍
            if (inputIdx >= 0 && inputIdx < n) {
                sum += input[inputIdx] * filter[i];  // 從全域記憶體讀取 filter（較慢）
            }
        }
        output[idx] = sum;
    }
}

// 使用常數記憶體的 1D 卷積（較快的版本）
// 注意：參數不需要傳入 filter，因為已經存在常數記憶體中
__global__ void convolutionConstant(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        float sum = 0.0f;
        int halfSize = FILTER_SIZE / 2;

        for (int i = 0; i < FILTER_SIZE; i++) {
            int inputIdx = idx - halfSize + i;
            if (inputIdx >= 0 && inputIdx < n) {
                // 從常數記憶體讀取 c_filter → 有快取加速，且所有執行緒讀同一個 i 時會廣播
                sum += input[inputIdx] * c_filter[i];
            }
        }
        output[idx] = sum;
    }
}

// 計時用的輔助函數
float runKernelGlobal(float *d_input, float *d_output, float *d_filter, int n) {
    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 暖機
    convolutionGlobal<<<blocks, threadsPerBlock>>>(d_input, d_output, d_filter, n);
    cudaDeviceSynchronize();

    // 計時
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        convolutionGlobal<<<blocks, threadsPerBlock>>>(d_input, d_output, d_filter, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds / 100.0f;
}

float runKernelConstant(float *d_input, float *d_output, int n) {
    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 暖機
    convolutionConstant<<<blocks, threadsPerBlock>>>(d_input, d_output, n);
    cudaDeviceSynchronize();

    // 計時
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        convolutionConstant<<<blocks, threadsPerBlock>>>(d_input, d_output, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds / 100.0f;
}

int main() {
    printf("========================================\n");
    printf("    常數記憶體 vs 全域記憶體 效能比較\n");
    printf("========================================\n\n");

    // 初始化濾波器（高斯模糊）
    float h_filter[FILTER_SIZE] = {
        0.05f, 0.1f, 0.15f, 0.2f, 0.2f, 0.15f, 0.1f, 0.05f, 0.0f
    };

    printf("濾波器大小: %d\n", FILTER_SIZE);
    printf("資料大小: %d 個元素\n\n", N);

    // 用 cudaMemcpyToSymbol 將濾波器複製到常數記憶體
    // ⚠️ 注意：不能用 cudaMemcpy，必須用 cudaMemcpyToSymbol 才能寫入 __constant__ 變數
    cudaMemcpyToSymbol(c_filter, h_filter, FILTER_SIZE * sizeof(float));

    // 分配記憶體
    float *h_input = (float*)malloc(N * sizeof(float));
    float *h_output = (float*)malloc(N * sizeof(float));

    // 初始化輸入資料
    srand(42);
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)(rand() % 100) / 100.0f;
    }

    // GPU 全域記憶體分配
    float *d_input, *d_output, *d_filter;
    cudaMalloc(&d_input, N * sizeof(float));              // 輸入資料
    cudaMalloc(&d_output, N * sizeof(float));             // 輸出結果
    cudaMalloc(&d_filter, FILTER_SIZE * sizeof(float));   // 全域記憶體版的 filter（對比用）

    // 將資料從 CPU（Host）複製到 GPU（Device）
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, FILTER_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // 執行測試
    printf("執行效能測試（每種方法執行 100 次取平均）...\n\n");

    float timeGlobal = runKernelGlobal(d_input, d_output, d_filter, N);
    float timeConstant = runKernelConstant(d_input, d_output, N);

    printf("結果：\n");
    printf("----------------------------------------\n");
    printf("全域記憶體版本:   %.4f ms\n", timeGlobal);
    printf("常數記憶體版本:   %.4f ms\n", timeConstant);
    printf("----------------------------------------\n");
    printf("加速比: %.2fx\n\n", timeGlobal / timeConstant);

    // 驗證結果正確性
    float *h_output_global = (float*)malloc(N * sizeof(float));
    float *h_output_constant = (float*)malloc(N * sizeof(float));

    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    convolutionGlobal<<<blocks, threadsPerBlock>>>(d_input, d_output, d_filter, N);
    cudaMemcpy(h_output_global, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    convolutionConstant<<<blocks, threadsPerBlock>>>(d_input, d_output, N);
    cudaMemcpy(h_output_constant, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    // 比較兩種方法的結果是否一致
    bool correct = true;
    for (int i = 0; i < N; i++) {
        // 💡 Debug 提示：浮點數比較要用容差（epsilon），不能用 ==
        if (fabsf(h_output_global[i] - h_output_constant[i]) > 1e-5) {
            correct = false;
            break;
        }
    }

    printf("結果驗證: %s\n\n", correct ? "通過" : "失敗");

    printf("關鍵概念：\n");
    printf("1. 常數記憶體大小限制為 64KB\n");
    printf("2. 當所有執行緒讀取相同位址時，廣播機制很有效\n");
    printf("3. 常數記憶體有專用的 L1 快取\n");
    printf("4. 適合存放濾波器係數、變換矩陣等\n");

    // 清理
    free(h_input);
    free(h_output);
    free(h_output_global);
    free(h_output_constant);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_filter);

    return 0;
}
