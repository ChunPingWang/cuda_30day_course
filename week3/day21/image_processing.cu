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

// CUDA 錯誤檢查巨集：包裝每個 CUDA API 呼叫，失敗時印出錯誤訊息並終止程式
// 💡 Debug 提示：正式程式中應該對每個 cudaMalloc、cudaMemcpy 都用這個巨集包起來
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", \
                cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// __constant__：常數記憶體，適合存放所有執行緒都會讀取的小型唯讀資料
__constant__ float d_gaussianKernel[25];  // 5x5 高斯模糊核心（25 個 float = 100 bytes）

// RGB 轉灰階（每個像素由一個執行緒處理）
// 使用 2D grid：blockIdx.x/y 和 threadIdx.x/y 分別對應圖像的 x/y 座標
__global__ void rgbToGray(unsigned char *input, unsigned char *output,
                          int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // 像素的 x 座標
    int y = blockIdx.y * blockDim.y + threadIdx.y;  // 像素的 y 座標

    if (x < width && y < height) {
        int grayIdx = y * width + x;       // 灰階圖的 1D 索引
        int rgbIdx = grayIdx * 3;          // RGB 圖的 1D 索引（每像素 3 bytes: R,G,B）

        // 人眼對綠色最敏感，紅色次之，藍色最不敏感 → 使用 ITU-R BT.601 標準加權
        float gray = 0.299f * input[rgbIdx] +       // Red
                     0.587f * input[rgbIdx + 1] +    // Green
                     0.114f * input[rgbIdx + 2];     // Blue

        output[grayIdx] = (unsigned char)gray;
    }
}

// 高斯模糊（5x5 卷積核心）— 基本版本，直接從全域記憶體讀取像素
__global__ void gaussianBlur(unsigned char *input, unsigned char *output,
                              int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float sum = 0.0f;

        // 遍歷 5x5 的鄰域（ky, kx 從 -2 到 +2）
        for (int ky = -2; ky <= 2; ky++) {
            for (int kx = -2; kx <= 2; kx++) {
                // clamp 邊界處理：超出圖像範圍的座標用邊界值代替
                int ix = min(max(x + kx, 0), width - 1);
                int iy = min(max(y + ky, 0), height - 1);
                int kidx = (ky + 2) * 5 + (kx + 2);  // 核心的 1D 索引
                sum += input[iy * width + ix] * d_gaussianKernel[kidx];  // 從常數記憶體讀取核心
            }
        }

        // 將結果限制在 [0, 255] 範圍內
        output[y * width + x] = (unsigned char)fminf(fmaxf(sum, 0.0f), 255.0f);
    }
}

// Sobel 邊緣檢測：用水平與垂直梯度找出圖像中的邊緣
__global__ void sobelEdge(unsigned char *input, unsigned char *output,
                          int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // 跳過邊界像素（Sobel 需要 3x3 鄰域，邊界無法計算）
    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        // Sobel X 核心：偵測垂直邊緣（左右差異）
        // [-1  0  1]
        // [-2  0  2]
        // [-1  0  1]
        float gx = -input[(y-1)*width + (x-1)] + input[(y-1)*width + (x+1)]
                   -2*input[y*width + (x-1)] + 2*input[y*width + (x+1)]
                   -input[(y+1)*width + (x-1)] + input[(y+1)*width + (x+1)];

        // Sobel Y 核心：偵測水平邊緣（上下差異）
        // [-1 -2 -1]
        // [ 0  0  0]
        // [ 1  2  1]
        float gy = -input[(y-1)*width + (x-1)] - 2*input[(y-1)*width + x] - input[(y-1)*width + (x+1)]
                   +input[(y+1)*width + (x-1)] + 2*input[(y+1)*width + x] + input[(y+1)*width + (x+1)];

        // 梯度強度 = sqrt(gx² + gy²)，值越大代表邊緣越明顯
        float magnitude = sqrtf(gx * gx + gy * gy);
        output[y * width + x] = (unsigned char)fminf(magnitude, 255.0f);
    } else if (x < width && y < height) {
        output[y * width + x] = 0;  // 邊界像素設為 0
    }
}

// 使用共享記憶體的優化版高斯模糊
// 優化原理：將像素先載入共享記憶體（tile），避免重複從慢速全域記憶體讀取
__global__ void gaussianBlurShared(unsigned char *input, unsigned char *output,
                                    int width, int height) {
    // tile 大小 = BLOCK_SIZE + 4（左右各多 2 像素的邊界，因為 5x5 核心半徑=2）
    __shared__ unsigned char tile[BLOCK_SIZE + 4][BLOCK_SIZE + 4];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;

    // 載入 tile（包含邊界的 halo 區域）
    int loadX = x - 2;  // 向左延伸 2 像素
    int loadY = y - 2;  // 向上延伸 2 像素

    // 每個執行緒可能需要載入多個像素來填滿 halo 區域
    for (int dy = 0; dy <= 4; dy += BLOCK_SIZE) {
        for (int dx = 0; dx <= 4; dx += BLOCK_SIZE) {
            int lx = tx + dx;
            int ly = ty + dy;
            if (lx < BLOCK_SIZE + 4 && ly < BLOCK_SIZE + 4) {
                int gx = min(max(loadX + dx, 0), width - 1);   // clamp 邊界
                int gy = min(max(loadY + dy, 0), height - 1);
                tile[ly][lx] = input[gy * width + gx];
            }
        }
    }

    __syncthreads();  // ⚠️ 注意：所有執行緒必須載入完畢後才能開始計算

    if (x < width && y < height) {
        float sum = 0.0f;

        // 直接從共享記憶體讀取 → 比全域記憶體快 10~100 倍
        for (int ky = 0; ky < 5; ky++) {
            for (int kx = 0; kx < 5; kx++) {
                sum += tile[ty + ky][tx + kx] * d_gaussianKernel[ky * 5 + kx];
            }
        }

        output[y * width + x] = (unsigned char)fminf(fmaxf(sum, 0.0f), 255.0f);
    }
}

// 在 CPU 端生成高斯核心（二維常態分布的離散化）
// sigma 控制模糊程度：sigma 越大越模糊
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

    // 分配 GPU 記憶體（d_ 前綴 = device，代表 GPU 端）
    unsigned char *d_rgb, *d_gray, *d_blur, *d_edge;
    CHECK_CUDA(cudaMalloc(&d_rgb, rgbSize));    // 原始 RGB 圖像
    CHECK_CUDA(cudaMalloc(&d_gray, graySize));  // 灰階結果
    CHECK_CUDA(cudaMalloc(&d_blur, graySize));  // 模糊結果
    CHECK_CUDA(cudaMalloc(&d_edge, graySize));  // 邊緣偵測結果

    // 將 RGB 圖像從 CPU 複製到 GPU
    CHECK_CUDA(cudaMemcpy(d_rgb, h_rgb, rgbSize, cudaMemcpyHostToDevice));

    // 設定高斯核心
    float h_gaussianKernel[25];
    generateGaussianKernel(h_gaussianKernel, 5, 1.5f);
    // 將高斯核心複製到 GPU 的常數記憶體
    cudaMemcpyToSymbol(d_gaussianKernel, h_gaussianKernel, 25 * sizeof(float));

    // 設定 2D 執行配置（因為圖像是二維的）
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);  // 每個 block 有 16x16=256 個執行緒
    dim3 blocks((width + BLOCK_SIZE - 1) / BLOCK_SIZE,    // x 方向需要的 block 數
                (height + BLOCK_SIZE - 1) / BLOCK_SIZE);   // y 方向需要的 block 數

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

    // 將處理結果從 GPU 複製回 CPU（cudaMemcpyDeviceToHost = GPU → CPU）
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

    // 清理：釋放所有 CPU 和 GPU 記憶體
    free(h_rgb);          // 釋放 CPU 記憶體
    free(h_gray);
    free(h_blur);
    free(h_edge);
    cudaFree(d_rgb);      // 釋放 GPU 記憶體
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
