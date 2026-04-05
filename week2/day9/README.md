# Day 9: 向量運算與最佳化

## 📚 今日學習目標

- 學習向量運算的最佳化技巧
- 理解記憶體合併（Memory Coalescing）基礎
- 學習使用向量資料類型
- 實作最佳化的向量運算

## 🚀 記憶體合併（Memory Coalescing）

### 什麼是記憶體合併？

當同一個 Warp 中的執行緒存取連續的記憶體位置時，GPU 可以將這些存取合併成較少的記憶體交易。

### 好的存取模式（合併）

```cuda
// 執行緒 0 存取 data[0]
// 執行緒 1 存取 data[1]
// 執行緒 2 存取 data[2]
// ...
int idx = threadIdx.x;
float value = data[idx];  // ✅ 連續存取 → 合併
```

### 差的存取模式（未合併）

```cuda
// 執行緒 0 存取 data[0]
// 執行緒 1 存取 data[32]
// 執行緒 2 存取 data[64]
// ...
int idx = threadIdx.x * 32;
float value = data[idx];  // ❌ 跳躍存取 → 未合併
```

### 效能差異

- 合併存取：1 次記憶體交易
- 未合併存取：可能需要 32 次記憶體交易
- 效能差距可達 **10-32 倍**！

## 📊 向量資料類型

CUDA 提供內建的向量類型，可以一次載入多個值：

### 可用類型

| 類型 | 元素數 | 總大小 |
|------|--------|--------|
| `float2` | 2 | 8 bytes |
| `float4` | 4 | 16 bytes |
| `int2` | 2 | 8 bytes |
| `int4` | 4 | 16 bytes |

### 使用範例

```cuda
__global__ void vectorAddFloat4(float4 *a, float4 *b, float4 *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        float4 va = a[idx];
        float4 vb = b[idx];

        float4 vc;
        vc.x = va.x + vb.x;
        vc.y = va.y + vb.y;
        vc.z = va.z + vb.z;
        vc.w = va.w + vb.w;

        c[idx] = vc;
    }
}
```

## 🔧 向量加法最佳化

### 版本 1：基本版本

```cuda
__global__ void vectorAddBasic(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
```

### 版本 2：使用 float4

```cuda
__global__ void vectorAddFloat4(float4 *a, float4 *b, float4 *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n / 4) {
        float4 va = a[idx];
        float4 vb = b[idx];
        c[idx] = make_float4(va.x + vb.x, va.y + vb.y,
                             va.z + vb.z, va.w + vb.w);
    }
}
```

### 版本 3：展開迴圈

```cuda
__global__ void vectorAddUnrolled(float *a, float *b, float *c, int n) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    if (idx + 3 < n) {
        c[idx]     = a[idx]     + b[idx];
        c[idx + 1] = a[idx + 1] + b[idx + 1];
        c[idx + 2] = a[idx + 2] + b[idx + 2];
        c[idx + 3] = a[idx + 3] + b[idx + 3];
    }
}
```

## 📈 效能比較

| 版本 | 特點 | 相對效能 |
|------|------|----------|
| 基本版 | 每個執行緒 1 個元素 | 1x |
| float4 | 每個執行緒 4 個元素 | ~1.5-2x |
| 展開迴圈 | 減少迴圈開銷 | ~1.3-1.5x |

## 💡 最佳化技巧總結

### 1. 確保記憶體合併
- 連續執行緒存取連續記憶體
- 避免跳躍存取

### 2. 使用向量類型
- 使用 `float2`、`float4` 等
- 減少記憶體交易次數

### 3. 迴圈展開
- 每個執行緒處理多個元素
- 減少指令開銷

### 4. 適當的 Block 大小
- 256 或 512 通常是好選擇
- 確保足夠的佔用率

## 🔧 實作練習

### 範例程式

1. **vector_add_optimized.cu** - 向量加法最佳化
2. **memory_patterns.cu** - 記憶體存取模式比較

## 🔧 編譯與執行

### CUDA 編譯

**Windows（cmd）：**

```cmd
nvcc -allow-unsupported-compiler -Wno-deprecated-gpu-targets -Xcompiler "/wd4819" -o vector_add_optimized.exe vector_add_optimized.cu
vector_add_optimized.exe
```

**Windows（PowerShell）：**

```powershell
nvcc -allow-unsupported-compiler -Wno-deprecated-gpu-targets -Xcompiler "/wd4819" -o vector_add_optimized.exe vector_add_optimized.cu
.\vector_add_optimized.exe
```

**WSL / Linux：**

```bash
nvcc -Wno-deprecated-gpu-targets -o vector_add_optimized vector_add_optimized.cu
./vector_add_optimized
```

### Python 等效

```python
import cupy as cp
n = 10_000_000
a = cp.random.rand(n, dtype=cp.float32)
b = cp.random.rand(n, dtype=cp.float32)
start = cp.cuda.Event(); end = cp.cuda.Event()
start.record()
c = a + b
end.record(); end.synchronize()
print(f"GPU 時間: {cp.cuda.get_elapsed_time(start, end):.2f} ms")
```

## 📝 今日作業

1. ✅ 理解記憶體合併的重要性
2. ✅ 執行向量加法的不同版本
3. ✅ 比較效能差異
4. ✅ 完成練習題

## 🎯 練習題

### 練習 1
將 Day 4 的向量加法改用 `float4` 實作。

### 練習 2
實作向量的 SAXPY 操作：`Y = a * X + Y`
使用向量類型最佳化。

---

**明天我們將進入矩陣運算！** 🎯
