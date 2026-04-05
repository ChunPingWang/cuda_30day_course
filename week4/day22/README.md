# Day 22: 影像處理實戰 - 模糊與邊緣檢測

## 📚 今日學習目標

- 實作完整的圖像模糊演算法
- 實作 Sobel 和 Canny 邊緣檢測
- 學習圖像處理的效能優化
- 使用真實圖片進行測試

## 🖼️ 今日專案

建立一個完整的圖像處理管線：

```
輸入圖片 → 灰度轉換 → 高斯模糊 → 邊緣檢測 → 輸出圖片
```

## 📝 主程式架構

```cuda
#include <stdio.h>
#include "stb_image.h"
#include "stb_image_write.h"

// 核心函數聲明
__global__ void rgbToGray(uchar4 *input, unsigned char *output,
                          int width, int height);
__global__ void gaussianBlur(unsigned char *input, unsigned char *output,
                             int width, int height);
__global__ void sobelEdge(unsigned char *input, unsigned char *output,
                          int width, int height);

int main(int argc, char *argv[]) {
    // 1. 讀取圖片
    int width, height, channels;
    unsigned char *h_image = stbi_load("input.jpg", &width, &height,
                                       &channels, 4);

    // 2. 分配 GPU 記憶體
    uchar4 *d_color;
    unsigned char *d_gray, *d_blur, *d_edge;
    size_t colorBytes = width * height * sizeof(uchar4);
    size_t grayBytes = width * height * sizeof(unsigned char);

    cudaMalloc(&d_color, colorBytes);
    cudaMalloc(&d_gray, grayBytes);
    cudaMalloc(&d_blur, grayBytes);
    cudaMalloc(&d_edge, grayBytes);

    // 3. 上傳圖片
    cudaMemcpy(d_color, h_image, colorBytes, cudaMemcpyHostToDevice);

    // 4. 處理管線
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    rgbToGray<<<grid, block>>>(d_color, d_gray, width, height);
    gaussianBlur<<<grid, block>>>(d_gray, d_blur, width, height);
    sobelEdge<<<grid, block>>>(d_blur, d_edge, width, height);

    // 5. 下載結果
    unsigned char *h_result = (unsigned char*)malloc(grayBytes);
    cudaMemcpy(h_result, d_edge, grayBytes, cudaMemcpyDeviceToHost);

    // 6. 儲存圖片
    stbi_write_png("output.png", width, height, 1, h_result, width);

    // 7. 清理
    // ...

    return 0;
}
```

## 🔧 實作練習

### 範例程式

1. **edge_detection.cu** - 完整的邊緣檢測
2. **blur_comparison.cu** - 不同模糊方法比較
3. **image_pipeline.cu** - 圖像處理管線

## 🔧 編譯與執行

### CUDA 編譯

**Windows（cmd）：**

```cmd
nvcc -allow-unsupported-compiler -Wno-deprecated-gpu-targets -Xcompiler "/wd4819" -o image_filters.exe image_filters.cu
image_filters.exe
```

**Windows（PowerShell）：**

```powershell
nvcc -allow-unsupported-compiler -Wno-deprecated-gpu-targets -Xcompiler "/wd4819" -o image_filters.exe image_filters.cu
.\image_filters.exe
```

**WSL / Linux：**

```bash
nvcc -Wno-deprecated-gpu-targets -o image_filters image_filters.cu
./image_filters
```

### Python 等效

```python
import cupy as cp
from cupyx.scipy.ndimage import convolve

img = cp.random.rand(512, 512).astype(cp.float32)

# Sobel 邊緣偵測
sobel_x = cp.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=cp.float32)
sobel_y = cp.array([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=cp.float32)
edges_x = convolve(img, sobel_x)
edges_y = convolve(img, sobel_y)
edges = cp.sqrt(edges_x**2 + edges_y**2)

# 銳化
sharpen = cp.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=cp.float32)
sharpened = convolve(img, sharpen)
```

---

**明天我們將學習直方圖計算！** 📊
