# Day 29: 期末專題（二）- 核心功能實作

## 今日學習目標

- 實作專題核心功能
- 整合各個模組
- 進行基本測試
- 除錯與修正

## 專題 A：圖像處理器 - 核心實作

### 1. 圖像結構與記憶體管理

```cuda
// src/image_io.cu
#include <stdio.h>
#include <stdlib.h>
#include "../include/image.h"

// 創建圖像
Image* createImage(int width, int height, int channels) {
    Image *img = (Image*)malloc(sizeof(Image));
    img->width = width;
    img->height = height;
    img->channels = channels;
    img->data = (unsigned char*)malloc(width * height * channels);
    return img;
}

// 釋放圖像
void freeImage(Image *img) {
    if (img) {
        free(img->data);
        free(img);
    }
}

// 創建 GPU 圖像
GpuImage* createGpuImage(int width, int height, int channels) {
    GpuImage *img = (GpuImage*)malloc(sizeof(GpuImage));
    img->width = width;
    img->height = height;
    img->channels = channels;
    cudaMalloc(&img->d_data, width * height * channels);
    return img;
}

// 釋放 GPU 圖像
void freeGpuImage(GpuImage *img) {
    if (img) {
        cudaFree(img->d_data);
        free(img);
    }
}

// 上傳到 GPU
void uploadImage(Image *src, GpuImage *dst) {
    int size = src->width * src->height * src->channels;
    cudaMemcpy(dst->d_data, src->data, size, cudaMemcpyHostToDevice);
}

// 下載到 CPU
void downloadImage(GpuImage *src, Image *dst) {
    int size = src->width * src->height * src->channels;
    cudaMemcpy(dst->data, src->d_data, size, cudaMemcpyDeviceToHost);
}
```

### 2. 濾波器核心實作

```cuda
// src/filters.cu
#include <stdio.h>
#include "../include/filters.h"

#define BLOCK_SIZE 16

// 常數記憶體存放卷積核心
__constant__ float d_kernel[25];  // 最大 5x5

// RGB 轉灰階
__global__ void rgbToGrayKernel(unsigned char *src, unsigned char *dst,
                                 int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        int srcIdx = idx * 3;

        // 加權平均（人眼感知）
        float gray = 0.299f * src[srcIdx] +
                     0.587f * src[srcIdx + 1] +
                     0.114f * src[srcIdx + 2];

        dst[idx] = (unsigned char)gray;
    }
}

void rgbToGray(GpuImage *src, GpuImage *dst) {
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((src->width + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (src->height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    rgbToGrayKernel<<<blocks, threads>>>(src->d_data, dst->d_data,
                                          src->width, src->height);
    cudaDeviceSynchronize();
}

// 通用卷積核心（灰階圖像）
__global__ void convolve2DKernel(unsigned char *src, unsigned char *dst,
                                  int width, int height, int kernelSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float sum = 0.0f;
        int halfK = kernelSize / 2;

        for (int ky = -halfK; ky <= halfK; ky++) {
            for (int kx = -halfK; kx <= halfK; kx++) {
                int ix = min(max(x + kx, 0), width - 1);
                int iy = min(max(y + ky, 0), height - 1);

                int kernelIdx = (ky + halfK) * kernelSize + (kx + halfK);
                sum += src[iy * width + ix] * d_kernel[kernelIdx];
            }
        }

        // 限制在 0-255 範圍
        sum = fminf(fmaxf(sum, 0.0f), 255.0f);
        dst[y * width + x] = (unsigned char)sum;
    }
}

// 高斯模糊
void gaussianBlur(GpuImage *src, GpuImage *dst, int kernelSize, float sigma) {
    // 生成高斯核心
    float *h_kernel = (float*)malloc(kernelSize * kernelSize * sizeof(float));
    int halfK = kernelSize / 2;
    float sum = 0.0f;

    for (int y = -halfK; y <= halfK; y++) {
        for (int x = -halfK; x <= halfK; x++) {
            float val = expf(-(x*x + y*y) / (2 * sigma * sigma));
            h_kernel[(y + halfK) * kernelSize + (x + halfK)] = val;
            sum += val;
        }
    }

    // 正規化
    for (int i = 0; i < kernelSize * kernelSize; i++) {
        h_kernel[i] /= sum;
    }

    // 複製到常數記憶體
    cudaMemcpyToSymbol(d_kernel, h_kernel, kernelSize * kernelSize * sizeof(float));

    // 執行卷積
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((src->width + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (src->height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    convolve2DKernel<<<blocks, threads>>>(src->d_data, dst->d_data,
                                           src->width, src->height, kernelSize);
    cudaDeviceSynchronize();

    free(h_kernel);
}

// Sobel 邊緣檢測
__global__ void sobelKernel(unsigned char *src, unsigned char *dst,
                            int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        // Sobel 運算子
        float gx = -src[(y-1)*width + (x-1)] + src[(y-1)*width + (x+1)]
                   -2*src[y*width + (x-1)] + 2*src[y*width + (x+1)]
                   -src[(y+1)*width + (x-1)] + src[(y+1)*width + (x+1)];

        float gy = -src[(y-1)*width + (x-1)] - 2*src[(y-1)*width + x] - src[(y-1)*width + (x+1)]
                   +src[(y+1)*width + (x-1)] + 2*src[(y+1)*width + x] + src[(y+1)*width + (x+1)];

        float magnitude = sqrtf(gx * gx + gy * gy);
        magnitude = fminf(magnitude, 255.0f);

        dst[y * width + x] = (unsigned char)magnitude;
    } else if (x < width && y < height) {
        dst[y * width + x] = 0;
    }
}

void sobelEdgeDetection(GpuImage *src, GpuImage *dst) {
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((src->width + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (src->height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    sobelKernel<<<blocks, threads>>>(src->d_data, dst->d_data,
                                      src->width, src->height);
    cudaDeviceSynchronize();
}

// 銳化
void sharpen(GpuImage *src, GpuImage *dst, float strength) {
    float h_kernel[9] = {
        0, -strength, 0,
        -strength, 1 + 4*strength, -strength,
        0, -strength, 0
    };

    cudaMemcpyToSymbol(d_kernel, h_kernel, 9 * sizeof(float));

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((src->width + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (src->height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    convolve2DKernel<<<blocks, threads>>>(src->d_data, dst->d_data,
                                           src->width, src->height, 3);
    cudaDeviceSynchronize();
}
```

### 3. 直方圖均衡化

```cuda
// 直方圖均衡化
__global__ void histogramKernel(unsigned char *data, unsigned int *hist,
                                 int width, int height) {
    __shared__ unsigned int localHist[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 初始化
    if (tid < 256) {
        localHist[tid] = 0;
    }
    __syncthreads();

    // 計算直方圖
    if (idx < width * height) {
        atomicAdd(&localHist[data[idx]], 1);
    }
    __syncthreads();

    // 合併到全域
    if (tid < 256) {
        atomicAdd(&hist[tid], localHist[tid]);
    }
}

__global__ void equalizeKernel(unsigned char *data, unsigned char *lut,
                                int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < width * height) {
        data[idx] = lut[data[idx]];
    }
}

void histogramEqualization(GpuImage *img) {
    int size = img->width * img->height;

    // 分配直方圖記憶體
    unsigned int *d_hist;
    cudaMalloc(&d_hist, 256 * sizeof(unsigned int));
    cudaMemset(d_hist, 0, 256 * sizeof(unsigned int));

    // 計算直方圖
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    histogramKernel<<<blocks, threads>>>(img->d_data, d_hist, img->width, img->height);

    // 複製到主機計算 CDF 和 LUT
    unsigned int h_hist[256];
    cudaMemcpy(h_hist, d_hist, 256 * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    // 計算 CDF
    unsigned int cdf[256];
    cdf[0] = h_hist[0];
    for (int i = 1; i < 256; i++) {
        cdf[i] = cdf[i-1] + h_hist[i];
    }

    // 計算 LUT
    unsigned char h_lut[256];
    unsigned int cdfMin = 0;
    for (int i = 0; i < 256; i++) {
        if (cdf[i] > 0) {
            cdfMin = cdf[i];
            break;
        }
    }

    for (int i = 0; i < 256; i++) {
        h_lut[i] = (unsigned char)(255.0f * (cdf[i] - cdfMin) / (size - cdfMin));
    }

    // 複製 LUT 到 GPU 並應用
    unsigned char *d_lut;
    cudaMalloc(&d_lut, 256);
    cudaMemcpy(d_lut, h_lut, 256, cudaMemcpyHostToDevice);

    equalizeKernel<<<blocks, threads>>>(img->d_data, d_lut, img->width, img->height);

    cudaFree(d_hist);
    cudaFree(d_lut);
}
```

### 4. 主程式

```cuda
// src/main.cu
#include <stdio.h>
#include "../include/image.h"
#include "../include/filters.h"

int main(int argc, char **argv) {
    printf("========================================\n");
    printf("    GPU 圖像處理器\n");
    printf("========================================\n\n");

    // 創建測試圖像（512x512 灰階）
    int width = 512, height = 512;
    Image *img = createImage(width, height, 1);

    // 生成測試圖案（漸層 + 雜訊）
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            img->data[y * width + x] = (x + y) % 256;
        }
    }

    // 創建 GPU 圖像
    GpuImage *d_src = createGpuImage(width, height, 1);
    GpuImage *d_dst = createGpuImage(width, height, 1);

    uploadImage(img, d_src);

    // 測試各種濾波器
    printf("測試高斯模糊...\n");
    gaussianBlur(d_src, d_dst, 5, 1.5f);
    printf("完成！\n");

    printf("測試 Sobel 邊緣檢測...\n");
    sobelEdgeDetection(d_src, d_dst);
    printf("完成！\n");

    printf("測試銳化...\n");
    sharpen(d_src, d_dst, 0.5f);
    printf("完成！\n");

    printf("測試直方圖均衡化...\n");
    histogramEqualization(d_src);
    printf("完成！\n");

    // 下載結果
    downloadImage(d_dst, img);

    printf("\n所有測試完成！\n");

    // 清理
    freeImage(img);
    freeGpuImage(d_src);
    freeGpuImage(d_dst);

    return 0;
}
```

## 專題 B：矩陣乘法核心實作

```cuda
// 矩陣乘法 - Tiled 版本
#define TILE_SIZE 16

__global__ void matrixMulTiled(float *A, float *B, float *C,
                                int M, int N, int K) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // 載入 tile
        if (row < M && t * TILE_SIZE + threadIdx.x < K)
            tileA[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && t * TILE_SIZE + threadIdx.y < K)
            tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // 計算
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

## 專題 C：神經網路核心實作

```cuda
// 全連接層前向傳播
__global__ void denseForwardKernel(float *input, float *weights, float *bias,
                                    float *output, int in_features, int out_features,
                                    int batch_size) {
    int batch = blockIdx.y;
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (out_idx < out_features && batch < batch_size) {
        float sum = bias[out_idx];
        for (int i = 0; i < in_features; i++) {
            sum += input[batch * in_features + i] * weights[out_idx * in_features + i];
        }
        output[batch * out_features + out_idx] = sum;
    }
}

// ReLU 激活
__global__ void reluKernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}

// Softmax
__global__ void softmaxKernel(float *data, int batch_size, int num_classes) {
    int batch = blockIdx.x;

    if (batch < batch_size) {
        float *row = data + batch * num_classes;

        // 找最大值（數值穩定性）
        float maxVal = row[0];
        for (int i = 1; i < num_classes; i++) {
            maxVal = fmaxf(maxVal, row[i]);
        }

        // 計算 exp 和總和
        float sum = 0.0f;
        for (int i = 0; i < num_classes; i++) {
            row[i] = expf(row[i] - maxVal);
            sum += row[i];
        }

        // 正規化
        for (int i = 0; i < num_classes; i++) {
            row[i] /= sum;
        }
    }
}
```

## 🔧 編譯與執行

本日為專題實作，無範例程式需要編譯。

## 今日作業

1. 完成你選擇專題的核心功能
2. 進行單元測試
3. 修正發現的問題

---

**明天是最後一天：優化、測試與總結！**
