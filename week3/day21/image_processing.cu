#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/**
 * Day 21: 週末專題 - 圖像處理
 *
 * 實作基本的圖像處理功能：
 * 1. 灰階轉換
 * 2. 高斯模糊
 * 3. Sobel 邊緣檢測
 */

#define BLOCK_SIZE 16

// 錯誤檢查
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", \
                cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// 常數記憶體存放卷積核心
__constant__ float d_gaussianKernel[25];  // 5x5 核心

// RGB 轉灰階
__global__ void rgbToGray(unsigned char *input, unsigned char *output,
                          int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int grayIdx = y * width + x;
        int rgbIdx = grayIdx * 3;

        // 加權平均
        float gray = 0.299f * input[rgbIdx] +
                     0.587f * input[rgbIdx + 1] +
                     0.114f * input[rgbIdx + 2];

        output[grayIdx] = (unsigned char)gray;
    }
}

// 高斯模糊（5x5）
__global__ void gaussianBlur(unsigned char *input, unsigned char *output,
                              int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float sum = 0.0f;

        for (int ky = -2; ky <= 2; ky++) {
            for (int kx = -2; kx <= 2; kx++) {
                int ix = min(max(x + kx, 0), width - 1);
                int iy = min(max(y + ky, 0), height - 1);
                int kidx = (ky + 2) * 5 + (kx + 2);
                sum += input[iy * width + ix] * d_gaussianKernel[kidx];
            }
        }

        output[y * width + x] = (unsigned char)fminf(fmaxf(sum, 0.0f), 255.0f);
    }
}

// Sobel 邊緣檢測
__global__ void sobelEdge(unsigned char *input, unsigned char *output,
                          int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        // Sobel X
        float gx = -input[(y-1)*width + (x-1)] + input[(y-1)*width + (x+1)]
                   -2*input[y*width + (x-1)] + 2*input[y*width + (x+1)]
                   -input[(y+1)*width + (x-1)] + input[(y+1)*width + (x+1)];

        // Sobel Y
        float gy = -input[(y-1)*width + (x-1)] - 2*input[(y-1)*width + x] - input[(y-1)*width + (x+1)]
                   +input[(y+1)*width + (x-1)] + 2*input[(y+1)*width + x] + input[(y+1)*width + (x+1)];

        float magnitude = sqrtf(gx * gx + gy * gy);
        output[y * width + x] = (unsigned char)fminf(magnitude, 255.0f);
    } else if (x < width && y < height) {
        output[y * width + x] = 0;
    }
}

// 使用共享記憶體的優化版高斯模糊
__global__ void gaussianBlurShared(unsigned char *input, unsigned char *output,
                                    int width, int height) {
    __shared__ unsigned char tile[BLOCK_SIZE + 4][BLOCK_SIZE + 4];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;

    // 載入 tile（包含邊界）
    int loadX = x - 2;
    int loadY = y - 2;

    // 每個執行緒可能需要載入多個像素
    for (int dy = 0; dy <= 4; dy += BLOCK_SIZE) {
        for (int dx = 0; dx <= 4; dx += BLOCK_SIZE) {
            int lx = tx + dx;
            int ly = ty + dy;
            if (lx < BLOCK_SIZE + 4 && ly < BLOCK_SIZE + 4) {
                int gx = min(max(loadX + dx, 0), width - 1);
                int gy = min(max(loadY + dy, 0), height - 1);
                tile[ly][lx] = input[gy * width + gx];
            }
        }
    }

    __syncthreads();

    if (x < width && y < height) {
        float sum = 0.0f;

        for (int ky = 0; ky < 5; ky++) {
            for (int kx = 0; kx < 5; kx++) {
                sum += tile[ty + ky][tx + kx] * d_gaussianKernel[ky * 5 + kx];
            }
        }

        output[y * width + x] = (unsigned char)fminf(fmaxf(sum, 0.0f), 255.0f);
    }
}

// 生成高斯核心
void generateGaussianKernel(float *kernel, int size, float sigma) {
    int half = size / 2;
    float sum = 0.0f;

    for (int y = -half; y <= half; y++) {
        for (int x = -half; x <= half; x++) {
            float val = expf(-(x*x + y*y) / (2 * sigma * sigma));
            kernel[(y + half) * size + (x + half)] = val;
            sum += val;
        }
    }

    // 正規化
    for (int i = 0; i < size * size; i++) {
        kernel[i] /= sum;
    }
}

int main() {
    printf("========================================\n");
    printf("    GPU 圖像處理週末專題\n");
    printf("========================================\n\n");

    // 模擬圖像大小
    int width = 1920;
    int height = 1080;
    printf("圖像大小: %d x %d\n\n", width, height);

    size_t rgbSize = width * height * 3;
    size_t graySize = width * height;

    // 分配主機記憶體
    unsigned char *h_rgb = (unsigned char*)malloc(rgbSize);
    unsigned char *h_gray = (unsigned char*)malloc(graySize);
    unsigned char *h_blur = (unsigned char*)malloc(graySize);
    unsigned char *h_edge = (unsigned char*)malloc(graySize);

    // 生成測試圖像（漸層 + 一些形狀）
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = (y * width + x) * 3;
            // 背景漸層
            h_rgb[idx] = (x * 255) / width;      // R
            h_rgb[idx + 1] = (y * 255) / height; // G
            h_rgb[idx + 2] = 128;                 // B

            // 加入一些矩形（測試邊緣檢測）
            if (x > 400 && x < 600 && y > 300 && y < 500) {
                h_rgb[idx] = 255;
                h_rgb[idx + 1] = 255;
                h_rgb[idx + 2] = 255;
            }
            if (x > 1000 && x < 1200 && y > 400 && y < 600) {
                h_rgb[idx] = 0;
                h_rgb[idx + 1] = 0;
                h_rgb[idx + 2] = 0;
            }
        }
    }

    // 分配 GPU 記憶體
    unsigned char *d_rgb, *d_gray, *d_blur, *d_edge;
    CHECK_CUDA(cudaMalloc(&d_rgb, rgbSize));
    CHECK_CUDA(cudaMalloc(&d_gray, graySize));
    CHECK_CUDA(cudaMalloc(&d_blur, graySize));
    CHECK_CUDA(cudaMalloc(&d_edge, graySize));

    // 複製到 GPU
    CHECK_CUDA(cudaMemcpy(d_rgb, h_rgb, rgbSize, cudaMemcpyHostToDevice));

    // 設定高斯核心
    float h_gaussianKernel[25];
    generateGaussianKernel(h_gaussianKernel, 5, 1.5f);
    cudaMemcpyToSymbol(d_gaussianKernel, h_gaussianKernel, 25 * sizeof(float));

    // 設定執行配置
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((width + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 1. RGB 轉灰階
    printf("1. RGB 轉灰階...\n");
    cudaEventRecord(start);
    rgbToGray<<<blocks, threads>>>(d_rgb, d_gray, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float timeGray;
    cudaEventElapsedTime(&timeGray, start, stop);
    printf("   完成！耗時: %.3f ms\n", timeGray);

    // 2. 高斯模糊
    printf("2. 高斯模糊（5x5）...\n");
    cudaEventRecord(start);
    gaussianBlur<<<blocks, threads>>>(d_gray, d_blur, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float timeBlur;
    cudaEventElapsedTime(&timeBlur, start, stop);
    printf("   完成！耗時: %.3f ms\n", timeBlur);

    // 3. Sobel 邊緣檢測
    printf("3. Sobel 邊緣檢測...\n");
    cudaEventRecord(start);
    sobelEdge<<<blocks, threads>>>(d_gray, d_edge, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float timeEdge;
    cudaEventElapsedTime(&timeEdge, start, stop);
    printf("   完成！耗時: %.3f ms\n", timeEdge);

    // 4. 優化版高斯模糊（使用共享記憶體）
    printf("4. 高斯模糊（共享記憶體優化）...\n");
    cudaEventRecord(start);
    gaussianBlurShared<<<blocks, threads>>>(d_gray, d_blur, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float timeBlurOpt;
    cudaEventElapsedTime(&timeBlurOpt, start, stop);
    printf("   完成！耗時: %.3f ms\n", timeBlurOpt);
    printf("   優化加速: %.2fx\n", timeBlur / timeBlurOpt);

    // 複製結果回主機
    CHECK_CUDA(cudaMemcpy(h_gray, d_gray, graySize, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_blur, d_blur, graySize, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_edge, d_edge, graySize, cudaMemcpyDeviceToHost));

    // 統計結果
    printf("\n結果統計:\n");
    int edgePixels = 0;
    for (int i = 0; i < width * height; i++) {
        if (h_edge[i] > 50) edgePixels++;
    }
    printf("  邊緣像素數量: %d (%.1f%%)\n", edgePixels, 100.0f * edgePixels / (width * height));

    printf("\n總處理時間: %.3f ms\n", timeGray + timeBlur + timeEdge);
    printf("處理速度: %.1f 百萬像素/秒\n",
           (width * height * 3) / ((timeGray + timeBlur + timeEdge) * 1000));

    // 清理
    free(h_rgb);
    free(h_gray);
    free(h_blur);
    free(h_edge);
    cudaFree(d_rgb);
    cudaFree(d_gray);
    cudaFree(d_blur);
    cudaFree(d_edge);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("\n========================================\n");
    printf("    專題完成！\n");
    printf("========================================\n");

    return 0;
}
