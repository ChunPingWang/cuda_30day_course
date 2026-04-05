# Day 13: 錯誤處理與除錯技巧

## 📚 今日學習目標

- 學習 CUDA 錯誤處理的最佳實踐
- 掌握常見錯誤的診斷方法
- 學習使用 cuda-memcheck 等除錯工具
- 理解常見的 CUDA 錯誤類型

## ⚠️ CUDA 錯誤處理

### 錯誤檢查巨集

```cuda
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s (error code: %d)\n", \
                cudaGetErrorString(err), err); \
        fprintf(stderr, "    at %s:%d\n", __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// 使用方式
CHECK_CUDA(cudaMalloc(&d_ptr, size));
CHECK_CUDA(cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice));
```

### 核心函數錯誤檢查

```cuda
myKernel<<<blocks, threads>>>(args);

// 檢查核心函數啟動錯誤
CHECK_CUDA(cudaGetLastError());

// 檢查執行時錯誤
CHECK_CUDA(cudaDeviceSynchronize());
```

## 🐛 常見錯誤類型

### 1. 記憶體存取錯誤

```cuda
// ❌ 越界存取
__global__ void badKernel(int *arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    arr[idx] = 0;  // 沒有邊界檢查！
}

// ✅ 正確做法
__global__ void goodKernel(int *arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        arr[idx] = 0;
    }
}
```

### 2. 共享記憶體問題

```cuda
// ❌ 共享記憶體未初始化就讀取
__global__ void badShared() {
    __shared__ float data[256];
    float val = data[threadIdx.x];  // 未初始化！
}

// ✅ 正確做法
__global__ void goodShared() {
    __shared__ float data[256];
    data[threadIdx.x] = someValue;
    __syncthreads();
    float val = data[threadIdx.x];
}
```

### 3. 同步問題

```cuda
// ❌ 條件分支中的 syncthreads
__global__ void badSync() {
    if (threadIdx.x < 16) {
        __syncthreads();  // 死鎖！其他執行緒不會到達
    }
}
```

## 🔧 除錯工具

### 1. cuda-memcheck

檢測記憶體錯誤：

```bash
cuda-memcheck ./your_program
```

### 2. compute-sanitizer（CUDA 11+）

更強大的檢測工具：

```bash
compute-sanitizer --tool memcheck ./your_program
compute-sanitizer --tool racecheck ./your_program
```

### 3. printf 除錯

```cuda
__global__ void debugKernel(int *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Thread %d: value = %d\n", idx, data[idx]);
}
```

### 4. NSight（圖形化工具）

NVIDIA 提供的專業除錯和效能分析工具。

## 📋 錯誤碼列表

| 錯誤碼 | 說明 |
|--------|------|
| `cudaErrorMemoryAllocation` | 記憶體分配失敗 |
| `cudaErrorInvalidValue` | 無效的參數 |
| `cudaErrorInvalidDevicePointer` | 無效的設備指標 |
| `cudaErrorInvalidMemcpyDirection` | 無效的複製方向 |
| `cudaErrorLaunchFailure` | 核心函數執行失敗 |
| `cudaErrorLaunchTimeout` | 執行超時 |

## 🔧 實作練習

### 範例程式

1. **error_handling.cu** - 錯誤處理示範
2. **debug_example.cu** - 除錯技巧範例

## 🔧 編譯與執行

### CUDA 編譯

**Windows（cmd）：**

```cmd
nvcc -allow-unsupported-compiler -Wno-deprecated-gpu-targets -Xcompiler "/wd4819" -o error_handling.exe error_handling.cu
error_handling.exe
```

**Windows（PowerShell）：**

```powershell
nvcc -allow-unsupported-compiler -Wno-deprecated-gpu-targets -Xcompiler "/wd4819" -o error_handling.exe error_handling.cu
.\error_handling.exe
```

**WSL / Linux：**

```bash
nvcc -Wno-deprecated-gpu-targets -o error_handling error_handling.cu
./error_handling
```

### Python 等效

```python
import cupy as cp
try:
    # 嘗試分配超大記憶體
    x = cp.zeros(10**12, dtype=cp.float32)
except cp.cuda.memory.OutOfMemoryError as e:
    print(f"GPU 記憶體不足: {e}")
except Exception as e:
    print(f"錯誤: {e}")
```

## 📝 今日作業

1. ✅ 在你的程式中加入完整的錯誤檢查
2. ✅ 使用 cuda-memcheck 檢查程式
3. ✅ 練習識別和修正常見錯誤

---

**明天是週末專題：完整的矩陣乘法優化！** 🎯
