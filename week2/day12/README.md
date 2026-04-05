# Day 12: 同步與協調（Synchronization）

## 📚 今日學習目標

- 深入理解 CUDA 同步機制
- 學習不同層級的同步方法
- 理解原子操作（Atomic Operations）
- 避免競爭條件（Race Condition）

## 🔄 同步的層級

### 1. Block 內同步：`__syncthreads()`

讓同一個 Block 內的所有執行緒等待。

```cuda
__global__ void example() {
    __shared__ float data[256];

    // 階段 1：寫入共享記憶體
    data[threadIdx.x] = computeValue();

    __syncthreads();  // 等待所有執行緒完成寫入

    // 階段 2：讀取共享記憶體
    float result = data[(threadIdx.x + 1) % 256];
}
```

### 2. Warp 內同步：`__syncwarp()`

讓同一個 Warp（32 執行緒）內同步。

```cuda
__global__ void warpExample() {
    // Warp 內同步（CUDA 9.0+）
    __syncwarp();
}
```

### 3. Grid 層級同步

Block 之間**無法直接同步**！需要：
- 結束核心函數
- 使用 Cooperative Groups（進階）

```cuda
// 方法 1：多個核心函數
kernel1<<<...>>>();
cudaDeviceSynchronize();  // 等待所有 Block 完成
kernel2<<<...>>>();
```

## ⚛️ 原子操作（Atomic Operations）

當多個執行緒需要更新同一個記憶體位置時，使用原子操作避免競爭。

### 常用原子操作

| 函數 | 操作 |
|------|------|
| `atomicAdd(addr, val)` | `*addr += val` |
| `atomicSub(addr, val)` | `*addr -= val` |
| `atomicMax(addr, val)` | `*addr = max(*addr, val)` |
| `atomicMin(addr, val)` | `*addr = min(*addr, val)` |
| `atomicExch(addr, val)` | 交換並返回舊值 |
| `atomicCAS(addr, compare, val)` | Compare-And-Swap |

### 範例：計數器

```cuda
// ❌ 錯誤：競爭條件
__global__ void badCounter(int *counter) {
    int temp = *counter;
    *counter = temp + 1;  // 多個執行緒同時讀寫！
}

// ✅ 正確：使用原子操作
__global__ void goodCounter(int *counter) {
    atomicAdd(counter, 1);
}
```

### 範例：直方圖

```cuda
__global__ void histogram(int *data, int *hist, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        int bin = data[idx];
        atomicAdd(&hist[bin], 1);
    }
}
```

## ⚠️ 競爭條件（Race Condition）

當多個執行緒同時存取相同記憶體，且至少一個是寫入時發生。

### 問題範例

```cuda
__shared__ float sum;

if (threadIdx.x == 0) sum = 0.0f;
__syncthreads();

// ❌ 競爭條件！
sum += myValue;  // 所有執行緒同時寫入
```

### 解決方案

```cuda
__shared__ float sum;
__shared__ float partialSum[256];

// 每個執行緒寫入自己的位置
partialSum[threadIdx.x] = myValue;
__syncthreads();

// 序列化歸約（或使用平行歸約）
if (threadIdx.x == 0) {
    sum = 0.0f;
    for (int i = 0; i < blockDim.x; i++) {
        sum += partialSum[i];
    }
}
```

## 🔧 實作練習

### 範例程式

1. **sync_demo.cu** - 同步機制示範
2. **atomic_ops.cu** - 原子操作範例
3. **histogram.cu** - 直方圖計算

## 🔧 編譯與執行

### CUDA 編譯

**Windows（cmd）：**

```cmd
nvcc -allow-unsupported-compiler -Wno-deprecated-gpu-targets -Xcompiler "/wd4819" -o atomic_ops.exe atomic_ops.cu
atomic_ops.exe
```

**Windows（PowerShell）：**

```powershell
nvcc -allow-unsupported-compiler -Wno-deprecated-gpu-targets -Xcompiler "/wd4819" -o atomic_ops.exe atomic_ops.cu
.\atomic_ops.exe
```

**WSL / Linux：**

```bash
nvcc -Wno-deprecated-gpu-targets -o atomic_ops atomic_ops.cu
./atomic_ops
```

### Python 等效

```python
import cupy as cp
# CuPy 透過 RawKernel 支援原子操作
kernel = cp.RawKernel(r'''
extern "C" __global__ void atomicAddDemo(int *counter) {
    atomicAdd(counter, 1);
}
''', 'atomicAddDemo')
counter = cp.zeros(1, dtype=cp.int32)
kernel((10,), (256,), (counter,))
print(f"計數器: {counter[0]}")  # 2560
```

## 📝 今日作業

1. ✅ 理解不同層級的同步
2. ✅ 學會使用原子操作
3. ✅ 識別競爭條件
4. ✅ 完成練習題

## 🎯 練習題

### 練習 1
實作一個計算陣列總和的核心函數（使用 atomicAdd）。

### 練習 2
修正一段有競爭條件的程式碼。

---

**明天我們將學習錯誤處理與除錯技巧！** 🐛
