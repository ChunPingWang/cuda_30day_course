# Day 18: 平行歸約（Parallel Reduction）

## 📚 今日學習目標

- 理解歸約操作的概念
- 學習多種平行歸約演算法
- 實作高效的求和、最大值運算
- 優化歸約效能

## 🔄 什麼是歸約？

歸約是將一組值組合成單一結果的操作：

```
輸入: [a₀, a₁, a₂, a₃, a₄, a₅, a₆, a₇]
              ↓ 歸約（求和）
輸出: a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇
```

常見歸約操作：
- **求和**：sum = a₀ + a₁ + ... + aₙ
- **最大值**：max = max(a₀, a₁, ..., aₙ)
- **最小值**：min = min(a₀, a₁, ..., aₙ)
- **乘積**：product = a₀ × a₁ × ... × aₙ

## 🌳 平行歸約演算法

### 版本 1：交錯配對（Interleaved）

```
Step 0: [0] [1] [2] [3] [4] [5] [6] [7]
         ↓   ↓   ↓   ↓
Step 1: [0+1]   [2+3]   [4+5]   [6+7]
           ↓       ↓
Step 2: [0+1+2+3]   [4+5+6+7]
               ↓
Step 3: [0+1+2+3+4+5+6+7]
```

```cuda
__global__ void reduce_interleaved(float *data, float *result, int n) {
    __shared__ float sdata[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < n) ? data[idx] : 0.0f;
    __syncthreads();

    // 交錯歸約
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        result[blockIdx.x] = sdata[0];
    }
}
```

### 版本 2：順序配對（Sequential）

避免 Warp Divergence：

```cuda
for (int s = 1; s < blockDim.x; s *= 2) {
    int index = 2 * s * tid;
    if (index < blockDim.x) {
        sdata[index] += sdata[index + s];
    }
    __syncthreads();
}
```

### 版本 3：Warp 層級優化

最後 32 個元素可以不用 __syncthreads：

```cuda
// Warp 內歸約（不需要同步）
if (tid < 32) {
    volatile float *vdata = sdata;
    vdata[tid] += vdata[tid + 32];
    vdata[tid] += vdata[tid + 16];
    vdata[tid] += vdata[tid + 8];
    vdata[tid] += vdata[tid + 4];
    vdata[tid] += vdata[tid + 2];
    vdata[tid] += vdata[tid + 1];
}
```

### 版本 4：使用 Warp Shuffle

CUDA 9.0+ 最高效的方法：

```cuda
__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void reduce_shuffle(float *data, float *result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (idx < n) ? data[idx] : 0.0f;

    // Warp 內歸約
    val = warpReduceSum(val);

    // Block 內歸約
    __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int warpId = threadIdx.x / 32;

    if (lane == 0) shared[warpId] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
    if (warpId == 0) val = warpReduceSum(val);

    if (threadIdx.x == 0) result[blockIdx.x] = val;
}
```

## 📊 效能比較

| 版本 | 特點 | 相對效能 |
|------|------|----------|
| 交錯配對 | 基本版本 | 1x |
| 順序配對 | 減少 divergence | 1.5x |
| Warp 優化 | 減少同步 | 2x |
| Warp Shuffle | 最高效 | 3x |

## 🔧 實作練習

### 範例程式

1. **reduction_basic.cu** - 基本歸約
2. **reduction_optimized.cu** - 優化版本
3. **reduction_warp.cu** - Warp Shuffle 版本

## 📝 今日作業

1. ✅ 實作求和歸約
2. ✅ 實作最大值歸約
3. ✅ 比較不同版本的效能

---

**明天我們將學習掃描（Scan/Prefix Sum）！** 📊
