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
__global__ void sharpenFilter(unsigned char *input, unsigned char *output,
                               int width, int height, float strength) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        // 銳化核心: [0, -s, 0; -s, 1+4s, -s; 0, -s, 0]
        float center = (1.0f + 4.0f * strength) * input[y * width + x];
        float neighbors = strength * (
            input[(y-1) * width + x] +
            input[(y+1) * width + x] +
            input[y * width + (x-1)] +
            input[y * width + (x+1)]
        );

        float result = center - neighbors;
        result = fminf(fmaxf(result, 0.0f), 255.0f);
        output[y * width + x] = (unsigned char)result;
    } else if (x < width && y < height) {
        output[y * width + x] = input[y * width + x];
    }
}

// 浮雕效果
__global__ void embossFilter(unsigned char *input, unsigned char *output,
                              int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

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

// 計算直方圖
__global__ void computeHistogram(unsigned char *image, unsigned int *histogram,
                                  int width, int height) {
    __shared__ unsigned int localHist[NUM_BINS];

    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int numThreads = blockDim.x * blockDim.y;

    // 初始化共享記憶體
    for (int i = tid; i < NUM_BINS; i += numThreads) {
        localHist[i] = 0;
    }
    __syncthreads();

    // 計算位置
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        unsigned char val = image[y * width + x];
        atomicAdd(&localHist[val], 1);
    }
    __syncthreads();

    // 合併到全域直方圖
    for (int i = tid; i < NUM_BINS; i += numThreads) {
        if (localHist[i] > 0) {
            atomicAdd(&histogram[i], localHist[i]);
        }
    }
}

// 應用查找表
__global__ void applyLUT(unsigned char *image, unsigned char *lut,
                         int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        image[idx] = lut[image[idx]];
    }
}

// 直方圖均衡化（主機端）
void histogramEqualization(unsigned char *d_image, int width, int height) {
    int numPixels = width * height;

    // 分配直方圖記憶體
    unsigned int *d_hist;
    CHECK_CUDA(cudaMalloc(&d_hist, NUM_BINS * sizeof(unsigned int)));
    CHECK_CUDA(cudaMemset(d_hist, 0, NUM_BINS * sizeof(unsigned int)));

    // 計算直方圖
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((width + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    computeHistogram<<<blocks, threads>>>(d_image, d_hist, width, height);

    // 複製直方圖到主機
    unsigned int h_hist[NUM_BINS];
    CHECK_CUDA(cudaMemcpy(h_hist, d_hist, NUM_BINS * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));

    // 計算 CDF
    unsigned int cdf[NUM_BINS];
    cdf[0] = h_hist[0];
    for (int i = 1; i < NUM_BINS; i++) {
        cdf[i] = cdf[i-1] + h_hist[i];
    }

    // 找到最小非零 CDF
    unsigned int cdfMin = 0;
    for (int i = 0; i < NUM_BINS; i++) {
        if (cdf[i] > 0) {
            cdfMin = cdf[i];
            break;
        }
    }

    // 創建查找表
    unsigned char h_lut[NUM_BINS];
    for (int i = 0; i < NUM_BINS; i++) {
        float val = (float)(cdf[i] - cdfMin) / (numPixels - cdfMin) * 255.0f;
        h_lut[i] = (unsigned char)fminf(fmaxf(val, 0.0f), 255.0f);
    }

    // 複製 LUT 到 GPU 並應用
    unsigned char *d_lut;
    CHECK_CUDA(cudaMalloc(&d_lut, NUM_BINS));
    CHECK_CUDA(cudaMemcpy(d_lut, h_lut, NUM_BINS, cudaMemcpyHostToDevice));

    applyLUT<<<blocks, threads>>>(d_image, d_lut, width, height);

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

    // GPU 記憶體
    unsigned char *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, size));
    CHECK_CUDA(cudaMalloc(&d_output, size));
    CHECK_CUDA(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((width + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 1. 銳化濾波器
    printf("1. 銳化濾波器...\n");
    cudaEventRecord(start);
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
