# Day 26: Unified Memory 進階使用

## 今日學習目標

- 深入理解 Unified Memory 的工作原理
- 學習記憶體預取和提示
- 掌握效能優化技巧
- 了解不同 GPU 架構的差異

## Unified Memory 回顧

### 基本概念

```cuda
float *data;
cudaMallocManaged(&data, size);

// CPU 和 GPU 都可以存取
data[0] = 1.0f;           // CPU 存取
kernel<<<...>>>(data);     // GPU 存取
```

### 底層機制

```
┌─────────┐                    ┌─────────┐
│   CPU   │                    │   GPU   │
│ Memory  │ ← ─ ─ ─ ─ ─ ─ ─ →  │ Memory  │
└────┬────┘    Page Migration   └────┬────┘
     │                                │
     └────────────────────────────────┘
              Unified Address
```

當存取發生時：
1. **頁面錯誤**：存取不在本地的頁面
2. **頁面遷移**：資料自動移動到存取者
3. **驅動處理**：CUDA 驅動管理這一切

## 記憶體提示（Memory Hints）

### cudaMemAdvise

```cuda
// 告訴系統資料主要由哪個處理器使用
cudaMemAdvise(ptr, size, advice, device);
```

### 提示類型

| 提示 | 說明 |
|------|------|
| `cudaMemAdviseSetReadMostly` | 資料是唯讀的 |
| `cudaMemAdviseSetPreferredLocation` | 偏好的存放位置 |
| `cudaMemAdviseSetAccessedBy` | 會被哪個處理器存取 |

### 範例

```cuda
float *data;
cudaMallocManaged(&data, size);

// 告訴系統資料偏好在 GPU 0 上
cudaMemAdvise(data, size, cudaMemAdviseSetPreferredLocation, 0);

// 告訴系統 CPU 也會存取這個資料
cudaMemAdvise(data, size, cudaMemAdviseSetAccessedBy, cudaCpuDeviceId);
```

## 記憶體預取（Prefetching）

### 顯式預取

```cuda
// 將資料預先移動到 GPU
cudaMemPrefetchAsync(data, size, deviceId, stream);

// 將資料預先移動回 CPU
cudaMemPrefetchAsync(data, size, cudaCpuDeviceId, stream);
```

### 為什麼需要預取？

```
沒有預取：
核心開始 → 頁面錯誤 → 等待資料遷移 → 繼續執行
          ↑                        ↑
        浪費時間                  浪費時間

有預取：
預取開始 → 核心開始 → 資料已經在 GPU → 直接執行
            ↑
         資料遷移與其他工作重疊
```

### 預取範例

```cuda
#include <stdio.h>

#define N (1 << 20)

__global__ void kernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f;
    }
}

int main() {
    float *data;
    int size = N * sizeof(float);

    cudaMallocManaged(&data, size);

    // 初始化（CPU）
    for (int i = 0; i < N; i++) {
        data[i] = 1.0f;
    }

    // 預取到 GPU
    cudaMemPrefetchAsync(data, size, 0, NULL);

    // 執行核心
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    kernel<<<blocks, threads>>>(data, N);

    // 預取回 CPU
    cudaMemPrefetchAsync(data, size, cudaCpuDeviceId, NULL);
    cudaDeviceSynchronize();

    // 驗證（CPU）
    printf("data[0] = %f\n", data[0]);

    cudaFree(data);
    return 0;
}
```

## 唯讀資料優化

```cuda
float *readOnlyData;
cudaMallocManaged(&readOnlyData, size);

// 初始化資料
initData(readOnlyData, N);

// 標記為唯讀 - 系統可以複製而非遷移
cudaMemAdvise(readOnlyData, size, cudaMemAdviseSetReadMostly, 0);

// 核心可以從本地快取讀取
kernel<<<...>>>(readOnlyData);
```

## 效能比較

```cuda
#include <stdio.h>

#define N (1 << 24)

__global__ void kernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = sqrtf(data[idx]);
    }
}

void benchmark(const char *name, float *data, int size, int n) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    // 暖機
    kernel<<<blocks, threads>>>(data, n);
    cudaDeviceSynchronize();

    // 計時
    cudaEventRecord(start);
    kernel<<<blocks, threads>>>(data, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("%s: %.3f ms\n", name, ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    int size = N * sizeof(float);

    // 方法 1：基本 Unified Memory
    float *data1;
    cudaMallocManaged(&data1, size);
    for (int i = 0; i < N; i++) data1[i] = (float)i;

    benchmark("基本 Unified Memory", data1, size, N);

    // 方法 2：使用預取
    float *data2;
    cudaMallocManaged(&data2, size);
    for (int i = 0; i < N; i++) data2[i] = (float)i;
    cudaMemPrefetchAsync(data2, size, 0, NULL);
    cudaDeviceSynchronize();

    benchmark("有預取的 Unified Memory", data2, size, N);

    // 方法 3：傳統方式
    float *h_data = (float*)malloc(size);
    float *d_data;
    cudaMalloc(&d_data, size);
    for (int i = 0; i < N; i++) h_data[i] = (float)i;
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    kernel<<<blocks, threads>>>(d_data, N);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    kernel<<<blocks, threads>>>(d_data, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("傳統 cudaMalloc: %.3f ms\n", ms);

    // 清理
    cudaFree(data1);
    cudaFree(data2);
    cudaFree(d_data);
    free(h_data);

    return 0;
}
```

## GPU 架構差異

### Pascal 之前（CC < 6.0）

- Unified Memory 只有基本支援
- 頁面遷移開銷較大
- 建議使用傳統記憶體管理

### Pascal 及之後（CC >= 6.0）

- 硬體支援頁面遷移
- 支援 Memory Oversubscription
- 效能更接近傳統方式

### Volta 及之後（CC >= 7.0）

- 更精細的頁面粒度
- 原子操作的改進
- 更好的多 GPU 支援

## 最佳實踐

### 1. 使用預取

```cuda
// 在核心執行前預取
cudaMemPrefetchAsync(data, size, gpuId, stream);
kernel<<<..., stream>>>(data);
```

### 2. 設定正確的提示

```cuda
// 主要在 GPU 使用
cudaMemAdvise(data, size, cudaMemAdviseSetPreferredLocation, gpuId);

// 唯讀資料
cudaMemAdvise(readOnly, size, cudaMemAdviseSetReadMostly, gpuId);
```

### 3. 避免頻繁的 CPU-GPU 切換

```cuda
// 不好：頻繁切換
for (int i = 0; i < 1000; i++) {
    kernel<<<...>>>(data);
    cudaDeviceSynchronize();
    processOnCPU(data);  // 觸發頁面遷移
}

// 好：批次處理
kernel<<<...>>>(data);
cudaMemPrefetchAsync(data, size, cudaCpuDeviceId);
cudaDeviceSynchronize();
processOnCPU(data);
```

## 今日作業

1. 比較有無預取的效能差異
2. 使用 `cudaMemAdvise` 優化唯讀資料存取
3. 在不同 GPU 上測試效能

## 🔧 編譯與執行

### CUDA 編譯

**Windows（cmd）：**

```cmd
nvcc -allow-unsupported-compiler -Wno-deprecated-gpu-targets -Xcompiler "/wd4819" -o unified_memory_advanced.exe unified_memory_advanced.cu
unified_memory_advanced.exe
```

**Windows（PowerShell）：**

```powershell
nvcc -allow-unsupported-compiler -Wno-deprecated-gpu-targets -Xcompiler "/wd4819" -o unified_memory_advanced.exe unified_memory_advanced.cu
.\unified_memory_advanced.exe
```

**WSL / Linux：**

```bash
nvcc -Wno-deprecated-gpu-targets -o unified_memory_advanced unified_memory_advanced.cu
./unified_memory_advanced
```

### Python 等效

```python
import cupy as cp
import numpy as np

# CuPy 自動管理 GPU 記憶體
a = cp.array([1, 2, 3, 4, 5], dtype=cp.float32)
b = a * 2  # GPU 上運算

# 與 NumPy 互操作（自動傳輸）
a_np = cp.asnumpy(b)          # GPU -> CPU
a_gpu = cp.asarray(a_np)      # CPU -> GPU

# 記憶體池資訊
pool = cp.get_default_memory_pool()
print(f"已使用 GPU 記憶體: {pool.used_bytes() / 1024:.1f} KB")
print(f"已分配 GPU 記憶體: {pool.total_bytes() / 1024:.1f} KB")
```

---

**明天我們將學習 Python 與 CUDA 的整合（CuPy/PyTorch）！**
