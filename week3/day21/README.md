# Day 21: 週末專題 - 圖像處理

## 📚 今日學習目標

- 整合第三週所學的優化技巧
- 實作多種圖像處理演算法
- 學習圖像資料在 GPU 上的處理方式
- 完成一個完整的圖像處理專案

## 🖼️ 專題內容

我們將實作以下圖像處理效果：

1. **亮度調整**（Brightness）
2. **灰度轉換**（Grayscale）
3. **高斯模糊**（Gaussian Blur）
4. **邊緣檢測**（Sobel Edge Detection）

## 📊 圖像資料結構

```cuda
// 圖像通常是 RGBA 格式，每個像素 4 bytes
struct Pixel {
    unsigned char r, g, b, a;
};

// 或使用 uchar4
uchar4 pixel;
pixel.x = r;  // 紅
pixel.y = g;  // 綠
pixel.z = b;  // 藍
pixel.w = a;  // 透明度
```

## 💡 核心函數實作

### 1. 亮度調整

```cuda
__global__ void adjustBrightness(uchar4 *image, int width, int height,
                                  int brightness) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        uchar4 pixel = image[idx];

        // 調整亮度並限制在 0-255
        int r = min(max(pixel.x + brightness, 0), 255);
        int g = min(max(pixel.y + brightness, 0), 255);
        int b = min(max(pixel.z + brightness, 0), 255);

        image[idx] = make_uchar4(r, g, b, pixel.w);
    }
}
```

### 2. 灰度轉換

```cuda
__global__ void toGrayscale(uchar4 *input, unsigned char *output,
                            int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        uchar4 pixel = input[idx];

        // 加權平均（人眼對綠色最敏感）
        unsigned char gray = (unsigned char)(
            0.299f * pixel.x +
            0.587f * pixel.y +
            0.114f * pixel.z
        );

        output[idx] = gray;
    }
}
```

### 3. 高斯模糊（使用共享記憶體）

```cuda
#define BLUR_SIZE 3

__global__ void gaussianBlur(unsigned char *input, unsigned char *output,
                             int width, int height) {
    __shared__ float tile[BLOCK_SIZE + 2][BLOCK_SIZE + 2];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // 載入資料（包含邊界）
    // ... 省略載入邏輯 ...

    __syncthreads();

    if (x < width && y < height) {
        // 3×3 高斯核
        float kernel[3][3] = {
            {1/16.0f, 2/16.0f, 1/16.0f},
            {2/16.0f, 4/16.0f, 2/16.0f},
            {1/16.0f, 2/16.0f, 1/16.0f}
        };

        float sum = 0.0f;
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                sum += tile[threadIdx.y + 1 + dy][threadIdx.x + 1 + dx]
                       * kernel[dy + 1][dx + 1];
            }
        }

        output[y * width + x] = (unsigned char)sum;
    }
}
```

### 4. Sobel 邊緣檢測

```cuda
__global__ void sobelEdge(unsigned char *input, unsigned char *output,
                          int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        // Sobel X 和 Y 核
        int gx = -input[(y-1)*width + (x-1)] + input[(y-1)*width + (x+1)]
                 -2*input[y*width + (x-1)] + 2*input[y*width + (x+1)]
                 -input[(y+1)*width + (x-1)] + input[(y+1)*width + (x+1)];

        int gy = -input[(y-1)*width + (x-1)] - 2*input[(y-1)*width + x]
                 -input[(y-1)*width + (x+1)]
                 +input[(y+1)*width + (x-1)] + 2*input[(y+1)*width + x]
                 +input[(y+1)*width + (x+1)];

        // 計算梯度大小
        int magnitude = min((int)sqrtf((float)(gx*gx + gy*gy)), 255);
        output[y * width + x] = (unsigned char)magnitude;
    }
}
```

## 🔧 專案結構

```
week3/day21/
├── README.md
├── image_processing.cu      # 主程式
├── kernels.cuh              # 核心函數
├── stb_image.h              # 圖片讀寫庫（header-only）
├── stb_image_write.h
├── input.jpg                # 測試圖片
└── Makefile
```

## 🔧 編譯與執行

### CUDA 編譯

**Windows（cmd）：**

```cmd
nvcc -allow-unsupported-compiler -Wno-deprecated-gpu-targets -Xcompiler "/wd4819" -o image_processing.exe image_processing.cu
image_processing.exe
```

**Windows（PowerShell）：**

```powershell
nvcc -allow-unsupported-compiler -Wno-deprecated-gpu-targets -Xcompiler "/wd4819" -o image_processing.exe image_processing.cu
.\image_processing.exe
```

**WSL / Linux：**

```bash
nvcc -Wno-deprecated-gpu-targets -o image_processing image_processing.cu
./image_processing
```

### Python 等效

```python
import cupy as cp
import numpy as np
# 模擬灰階影像
img = cp.random.randint(0, 256, (512, 512), dtype=cp.uint8).astype(cp.float32)

# 高斯模糊（3x3 核心）
kernel = cp.array([[1,2,1],[2,4,2],[1,2,1]], dtype=cp.float32) / 16
from cupyx.scipy.ndimage import convolve
blurred = convolve(img, kernel)

# 亮度調整
brightened = cp.clip(img * 1.2, 0, 255)
```

## 🎯 效能優化

1. **使用共享記憶體**：減少重複的全域記憶體讀取
2. **合併記憶體存取**：確保連續執行緒存取連續像素
3. **使用 uchar4**：一次載入 4 個 bytes
4. **流水線處理**：使用 CUDA Streams 並行處理多張圖片

## 📝 本週總結

### 學到的技術

1. **記憶體合併**優化
2. **Bank Conflict** 避免
3. **平行歸約**演算法
4. **掃描/前綴和**演算法
5. **效能分析**工具使用

---

**恭喜完成第三週！🎉 下週是實戰應用週！**
