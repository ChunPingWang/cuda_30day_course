# Day 16: 紋理記憶體與常數記憶體

## 今日學習目標

- 了解紋理記憶體（Texture Memory）的特性
- 學習常數記憶體（Constant Memory）的使用
- 理解這些特殊記憶體的最佳使用情境
- 實作使用特殊記憶體的範例

## 記憶體類型回顧

| 記憶體類型 | 位置 | 快取 | 存取權限 | 生命週期 |
|-----------|------|------|----------|----------|
| 全域記憶體 | GPU | L2 | 讀/寫 | 應用程式 |
| 共享記憶體 | SM | - | 讀/寫 | Block |
| **常數記憶體** | GPU | L1 | 唯讀 | 應用程式 |
| **紋理記憶體** | GPU | L1 | 唯讀 | 應用程式 |
| 暫存器 | SM | - | 讀/寫 | 執行緒 |

## 常數記憶體（Constant Memory）

### 特性

- 大小限制：64 KB
- 對所有執行緒唯讀
- 有專用快取（每個 SM 8 KB）
- **廣播機制**：同一 Warp 讀取相同位址時只需一次存取

### 最佳使用情境

1. 所有執行緒讀取相同資料
2. 卷積核心（Convolution Kernels）
3. 查找表（Look-up Tables）
4. 物理常數

### 宣告與使用

```cuda
#include <stdio.h>

// 在全域範圍宣告常數記憶體
__constant__ float constCoeffs[256];

__global__ void applyFilter(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float sum = 0.0f;
        // 所有執行緒讀取相同的係數 - 廣播！
        for (int i = 0; i < 5; i++) {
            if (idx + i < n) {
                sum += input[idx + i] * constCoeffs[i];
            }
        }
        output[idx] = sum;
    }
}

int main() {
    // 在主機端初始化係數
    float h_coeffs[5] = {0.1f, 0.2f, 0.4f, 0.2f, 0.1f};

    // 複製到常數記憶體
    cudaMemcpyToSymbol(constCoeffs, h_coeffs, 5 * sizeof(float));

    // ... 執行核心函數
    return 0;
}
```

### 效能比較

```cuda
// 使用全域記憶體（較慢）
__global__ void filterGlobal(float *input, float *output,
                              float *coeffs, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float sum = 0.0f;
        for (int i = 0; i < 5; i++) {
            sum += input[idx + i] * coeffs[i];  // 每次都從全域記憶體讀取
        }
        output[idx] = sum;
    }
}

// 使用常數記憶體（較快）
__constant__ float c_coeffs[5];

__global__ void filterConstant(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float sum = 0.0f;
        for (int i = 0; i < 5; i++) {
            sum += input[idx + i] * c_coeffs[i];  // 從快取讀取，且可廣播
        }
        output[idx] = sum;
    }
}
```

## 紋理記憶體（Texture Memory）

### 特性

- 針對 2D 空間局部性優化
- 自動處理邊界條件
- 支援硬體插值
- 有專用快取

### 最佳使用情境

1. 2D 圖像處理
2. 非規則存取模式
3. 需要邊界處理的運算
4. 需要插值的應用

### 現代 CUDA 紋理物件 API

```cuda
#include <stdio.h>

// 使用紋理物件 API（CUDA 5.0+）
__global__ void textureKernel(cudaTextureObject_t texObj,
                               float *output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        // 使用正規化座標讀取紋理
        float u = (x + 0.5f) / width;
        float v = (y + 0.5f) / height;

        // 讀取紋理值（自動插值）
        float value = tex2D<float>(texObj, u, v);
        output[y * width + x] = value;
    }
}

int main() {
    int width = 512, height = 512;
    size_t size = width * height * sizeof(float);

    // 分配和初始化資料
    float *h_data = (float*)malloc(size);
    for (int i = 0; i < width * height; i++) {
        h_data[i] = (float)i;
    }

    // 創建 CUDA 陣列
    cudaChannelFormatDesc channelDesc =
        cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaArray_t cuArray;
    cudaMallocArray(&cuArray, &channelDesc, width, height);

    // 複製資料到 CUDA 陣列
    cudaMemcpy2DToArray(cuArray, 0, 0, h_data, width * sizeof(float),
                        width * sizeof(float), height, cudaMemcpyHostToDevice);

    // 設定紋理資源描述
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    // 設定紋理描述
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;  // 邊界處理
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;       // 線性插值
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;                    // 正規化座標

    // 創建紋理物件
    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

    // 分配輸出記憶體
    float *d_output;
    cudaMalloc(&d_output, size);

    // 執行核心
    dim3 threads(16, 16);
    dim3 blocks((width + 15) / 16, (height + 15) / 16);
    textureKernel<<<blocks, threads>>>(texObj, d_output, width, height);

    // 清理
    cudaDestroyTextureObject(texObj);
    cudaFreeArray(cuArray);
    cudaFree(d_output);
    free(h_data);

    return 0;
}
```

## 邊界處理模式

紋理記憶體支援多種邊界處理模式：

| 模式 | 說明 |
|------|------|
| `cudaAddressModeClamp` | 夾緊到邊界值 |
| `cudaAddressModeWrap` | 環繞（重複） |
| `cudaAddressModeMirror` | 鏡像 |
| `cudaAddressModeBorder` | 返回邊界顏色（0） |

```
原始資料: [A][B][C][D]

Clamp:  [A][A][B][C][D][D][D]
Wrap:   [C][D][A][B][C][D][A]
Mirror: [B][A][A][B][C][D][D]
Border: [0][0][A][B][C][D][0]
```

## 實作範例：圖像模糊

```cuda
#include <stdio.h>

__constant__ float blurKernel[9];  // 3x3 模糊核心

__global__ void blurImage(cudaTextureObject_t texObj, float *output,
                          int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float sum = 0.0f;
        int kernelIdx = 0;

        // 3x3 卷積
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                float u = (x + dx + 0.5f) / width;
                float v = (y + dy + 0.5f) / height;
                sum += tex2D<float>(texObj, u, v) * blurKernel[kernelIdx++];
            }
        }

        output[y * width + x] = sum;
    }
}

int main() {
    // 初始化模糊核心（常數記憶體）
    float h_kernel[9] = {
        1.0f/9, 1.0f/9, 1.0f/9,
        1.0f/9, 1.0f/9, 1.0f/9,
        1.0f/9, 1.0f/9, 1.0f/9
    };
    cudaMemcpyToSymbol(blurKernel, h_kernel, 9 * sizeof(float));

    printf("圖像模糊範例：使用常數記憶體存放核心，紋理記憶體存放圖像\n");

    // ... 完整實作請見 texture_blur.cu

    return 0;
}
```

## 何時使用哪種記憶體？

### 使用常數記憶體

- 資料量 <= 64 KB
- 所有執行緒讀取相同資料
- 例如：濾波器係數、轉換矩陣

### 使用紋理記憶體

- 2D/3D 空間局部性存取
- 需要硬體插值
- 需要自動邊界處理
- 例如：圖像處理、體積渲染

### 使用共享記憶體

- 執行緒間需要共享資料
- 頻繁重用的資料
- 例如：矩陣乘法的 Tile

### 使用全域記憶體

- 大量資料
- 需要讀寫
- 無特殊存取模式

## 今日作業

1. 實作使用常數記憶體的 1D 卷積
2. 比較全域記憶體和常數記憶體的效能差異
3. 嘗試不同的紋理邊界處理模式

## 🔧 編譯與執行

### CUDA 編譯

**Windows（cmd）：**

```cmd
nvcc -allow-unsupported-compiler -Wno-deprecated-gpu-targets -Xcompiler "/wd4819" -o constant_memory.exe constant_memory.cu
constant_memory.exe
```

**Windows（PowerShell）：**

```powershell
nvcc -allow-unsupported-compiler -Wno-deprecated-gpu-targets -Xcompiler "/wd4819" -o constant_memory.exe constant_memory.cu
.\constant_memory.exe
```

**WSL / Linux：**

```bash
nvcc -Wno-deprecated-gpu-targets -o constant_memory constant_memory.cu
./constant_memory
```

### Python 等效

```python
import cupy as cp
# CuPy 中使用常數記憶體的 RawKernel
kernel = cp.RawKernel(r'''
__constant__ float coeff[5];
extern "C" __global__ void applyFilter(float *data, float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = data[idx] * coeff[0];
    }
}
''', 'applyFilter')
```

---

**明天我們將學習掃描（Scan）演算法！**
