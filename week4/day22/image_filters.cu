#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/**
 * Day 22: 進階圖像處理
 *
 * 實作更多圖像處理濾波器：
 * 1. 銳化濾波器
 * 2. 浮雕效果
 * 3. 直方圖均衡化
 */

#define BLOCK_SIZE 16
#define NUM_BINS 256

// 錯誤檢查
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// 銳化濾波器
// __global__ 表示這是一個由 GPU 執行的核心函式（kernel），由 CPU 端呼叫
__global__ void sharpenFilter(unsigned char *input, unsigned char *output,
                               int width, int height, float strength) {
    // blockIdx = 此執行緒所在的 block 編號，threadIdx = block 內的執行緒編號
    // blockDim = 每個 block 的執行緒數量；組合起來算出全域的 (x, y) 座標
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // ⚠️ 注意：邊界檢查很重要，避免讀取圖像外的記憶體
    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        // 銳化核心: [0, -s, 0; -s, 1+4s, -s; 0, -s, 0]
        // 把中心像素放大、減去鄰居像素，讓邊緣更明顯
        float center = (1.0f + 4.0f * strength) * input[y * width + x];
        float neighbors = strength * (
            input[(y-1) * width + x] +     // 上
            input[(y+1) * width + x] +     // 下
            input[y * width + (x-1)] +     // 左
            input[y * width + (x+1)]       // 右
        );

        float result = center - neighbors;
        // 💡 Debug 提示：如果結果圖像全黑或全白，檢查 strength 值是否太大
        result = fminf(fmaxf(result, 0.0f), 255.0f); // 限制在 0~255 範圍
        output[y * width + x] = (unsigned char)result;
    } else if (x < width && y < height) {
        output[y * width + x] = input[y * width + x]; // 邊界像素直接複製
    }
}

// 浮雕效果：讓圖像看起來像立體浮雕
__global__ void embossFilter(unsigned char *input, unsigned char *output,
                              int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; // 計算此執行緒負責的像素 x 座標
    int y = blockIdx.y * blockDim.y + threadIdx.y; // 計算此執行緒負責的像素 y 座標

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        // 浮雕核心: [-2, -1, 0; -1, 1, 1; 0, 1, 2]
        float result = -2.0f * input[(y-1) * width + (x-1)]
                       - input[(y-1) * width + x]
                       - input[y * width + (x-1)]
                       + input[y * width + x]
                       + input[y * width + (x+1)]
                       + input[(y+1) * width + x]
                       + 2.0f * input[(y+1) * width + (x+1)];

        // 偏移到中間灰度
        result = result + 128.0f;
        result = fminf(fmaxf(result, 0.0f), 255.0f);
        output[y * width + x] = (unsigned char)result;
    } else if (x < width && y < height) {
        output[y * width + x] = 128;  // 邊界設為中間灰度
    }
}

// 計算直方圖：統計每個灰階值 (0~255) 出現的次數
__global__ void computeHistogram(unsigned char *image, unsigned int *histogram,
                                  int width, int height) {
    // __shared__ 宣告共享記憶體，同一個 block 內的所有執行緒可以共用，速度比全域記憶體快很多
    __shared__ unsigned int localHist[NUM_BINS];

    // 把 2D 的 threadIdx 轉成 1D 的索引
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int numThreads = blockDim.x * blockDim.y;

    // 初始化共享記憶體（多個執行緒分工歸零）
    for (int i = tid; i < NUM_BINS; i += numThreads) {
        localHist[i] = 0;
    }
    // __syncthreads() 確保 block 內所有執行緒都完成初始化後才繼續
    // ⚠️ 注意：如果忘記這行，可能會讀到尚未歸零的值
    __syncthreads();

    // 計算位置
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        unsigned char val = image[y * width + x];
        // atomicAdd 確保多個執行緒同時累加時不會衝突（原子操作）
        atomicAdd(&localHist[val], 1);
    }
    __syncthreads(); // 等所有執行緒都統計完本地直方圖

    // 合併到全域直方圖（每個 block 的結果加到全域）
    for (int i = tid; i < NUM_BINS; i += numThreads) {
        if (localHist[i] > 0) {
            atomicAdd(&histogram[i], localHist[i]);
        }
    }
}

// 應用查找表（LUT, Look-Up Table）：用預先算好的對照表來快速轉換像素值
__global__ void applyLUT(unsigned char *image, unsigned char *lut,
                         int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        image[idx] = lut[image[idx]];
    }
}

// 直方圖均衡化（主機端）：讓低對比度的圖像變得更清晰
// 原理：重新分配灰階值，讓每個灰階值的像素數量更均勻
void histogramEqualization(unsigned char *d_image, int width, int height) {
    int numPixels = width * height;

    // cudaMalloc 在 GPU 上分配記憶體（類似 CPU 的 malloc）
    unsigned int *d_hist;
    CHECK_CUDA(cudaMalloc(&d_hist, NUM_BINS * sizeof(unsigned int)));
    CHECK_CUDA(cudaMemset(d_hist, 0, NUM_BINS * sizeof(unsigned int)));

    // 計算直方圖
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE); // 每個 block 有 16x16 = 256 個執行緒
    dim3 blocks((width + BLOCK_SIZE - 1) / BLOCK_SIZE,  // 計算需要多少 block 才能覆蓋整張圖
                (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // <<<blocks, threads>>> 是 CUDA 的核心啟動語法，指定 grid 和 block 的大小
    computeHistogram<<<blocks, threads>>>(d_image, d_hist, width, height);

    // cudaMemcpy 在 CPU 和 GPU 之間複製資料
    // cudaMemcpyDeviceToHost = 從 GPU 複製到 CPU
    unsigned int h_hist[NUM_BINS];
    CHECK_CUDA(cudaMemcpy(h_hist, d_hist, NUM_BINS * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));

    // 計算 CDF（累積分佈函式）：每個灰階值以下的像素累積數量
    unsigned int cdf[NUM_BINS];
    cdf[0] = h_hist[0];
    for (int i = 1; i < NUM_BINS; i++) {
        cdf[i] = cdf[i-1] + h_hist[i];
    }

    // 找到最小非零 CDF（用於均衡化公式）
    unsigned int cdfMin = 0;
    for (int i = 0; i < NUM_BINS; i++) {
        if (cdf[i] > 0) {
            cdfMin = cdf[i];
            break;
        }
    }

    // 創建查找表：將舊灰階值對應到新的均衡化灰階值
    unsigned char h_lut[NUM_BINS];
    for (int i = 0; i < NUM_BINS; i++) {
        // 💡 Debug 提示：如果均衡化後圖像全白，檢查 numPixels 是否 == cdfMin（除以零的情況）
        float val = (float)(cdf[i] - cdfMin) / (numPixels - cdfMin) * 255.0f;
        h_lut[i] = (unsigned char)fminf(fmaxf(val, 0.0f), 255.0f);
    }

    // 複製 LUT 到 GPU 並應用
    unsigned char *d_lut;
    CHECK_CUDA(cudaMalloc(&d_lut, NUM_BINS));
    // cudaMemcpyHostToDevice = 從 CPU 複製到 GPU
    CHECK_CUDA(cudaMemcpy(d_lut, h_lut, NUM_BINS, cudaMemcpyHostToDevice));

    applyLUT<<<blocks, threads>>>(d_image, d_lut, width, height);

    // cudaFree 釋放 GPU 記憶體（類似 CPU 的 free）
    // ⚠️ 注意：忘記 cudaFree 會造成 GPU 記憶體洩漏
    cudaFree(d_hist);
    cudaFree(d_lut);
}

int main() {
    printf("========================================\n");
    printf("    進階圖像處理濾波器\n");
    printf("========================================\n\n");

    int width = 1024;
    int height = 768;
    size_t size = width * height;

    printf("圖像大小: %d x %d\n\n", width, height);

    // 分配記憶體
    unsigned char *h_input = (unsigned char*)malloc(size);
    unsigned char *h_output = (unsigned char*)malloc(size);

    // 生成測試圖像（低對比度）
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // 壓縮對比度的灰階值
            int val = (x + y) % 128 + 64;  // 範圍 64-192

            // 加入一些圖案
            if ((x / 64 + y / 64) % 2 == 0) {
                val += 30;
            }

            h_input[y * width + x] = (unsigned char)val;
        }
    }

    // GPU 記憶體：變數名前綴 d_ 代表 device（GPU），h_ 代表 host（CPU）
    unsigned char *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, size));   // 在 GPU 分配記憶體
    CHECK_CUDA(cudaMalloc(&d_output, size));
    CHECK_CUDA(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice)); // CPU → GPU

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((width + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // cudaEvent 用來精確測量 GPU 上的執行時間
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 1. 銳化濾波器
    printf("1. 銳化濾波器...\n");
    cudaEventRecord(start);
    // <<<blocks, threads>>> 啟動 GPU kernel，blocks 個 block，每個 block 有 threads 個執行緒
    sharpenFilter<<<blocks, threads>>>(d_input, d_output, width, height, 1.0f);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float timeSharpen;
    cudaEventElapsedTime(&timeSharpen, start, stop);
    printf("   完成！耗時: %.3f ms\n", timeSharpen);

    // 2. 浮雕效果
    printf("2. 浮雕效果...\n");
    cudaEventRecord(start);
    embossFilter<<<blocks, threads>>>(d_input, d_output, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float timeEmboss;
    cudaEventElapsedTime(&timeEmboss, start, stop);
    printf("   完成！耗時: %.3f ms\n", timeEmboss);

    // 3. 直方圖均衡化
    printf("3. 直方圖均衡化...\n");
    CHECK_CUDA(cudaMemcpy(d_output, d_input, size, cudaMemcpyDeviceToDevice));

    cudaEventRecord(start);
    histogramEqualization(d_output, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float timeHist;
    cudaEventElapsedTime(&timeHist, start, stop);
    printf("   完成！耗時: %.3f ms\n", timeHist);

    // 統計
    CHECK_CUDA(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));

    // 計算對比度變化
    int minBefore = 255, maxBefore = 0;
    int minAfter = 255, maxAfter = 0;

    for (int i = 0; i < (int)size; i++) {
        if (h_input[i] < minBefore) minBefore = h_input[i];
        if (h_input[i] > maxBefore) maxBefore = h_input[i];
        if (h_output[i] < minAfter) minAfter = h_output[i];
        if (h_output[i] > maxAfter) maxAfter = h_output[i];
    }

    printf("\n對比度變化:\n");
    printf("  原始: %d - %d (範圍 %d)\n", minBefore, maxBefore, maxBefore - minBefore);
    printf("  均衡後: %d - %d (範圍 %d)\n", minAfter, maxAfter, maxAfter - minAfter);

    printf("\n總處理時間: %.3f ms\n", timeSharpen + timeEmboss + timeHist);

    // 清理
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("\n專題完成！\n");

    return 0;
}
