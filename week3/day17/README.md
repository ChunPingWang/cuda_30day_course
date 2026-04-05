# Day 17: 掃描（Scan / Prefix Sum）演算法

## 今日學習目標

- 理解掃描（Prefix Sum）的概念
- 學習 Inclusive 和 Exclusive Scan 的區別
- 實作高效的平行掃描演算法
- 了解掃描在實際應用中的重要性

## 什麼是掃描（Scan）？

掃描是一個將二元運算子應用到序列的前綴操作。

### Inclusive Scan（包含掃描）

每個輸出元素包含當前元素。

```
輸入:   [3, 1, 7, 0, 4, 1, 6, 3]
輸出:   [3, 4, 11, 11, 15, 16, 22, 25]

計算過程（使用加法）：
output[0] = 3
output[1] = 3 + 1 = 4
output[2] = 3 + 1 + 7 = 11
output[3] = 3 + 1 + 7 + 0 = 11
...
```

### Exclusive Scan（排除掃描）

每個輸出元素不包含當前元素。

```
輸入:   [3, 1, 7, 0, 4, 1, 6, 3]
輸出:   [0, 3, 4, 11, 11, 15, 16, 22]

計算過程：
output[0] = 0（identity）
output[1] = 3
output[2] = 3 + 1 = 4
output[3] = 3 + 1 + 7 = 11
...
```

## 為什麼掃描很重要？

掃描是許多平行演算法的基礎：

1. **排序**：Radix Sort
2. **壓縮**：Stream Compaction
3. **直方圖**：Histogram
4. **樹遍歷**：平行 BFS
5. **多項式計算**

## 循序版本

```c
// O(n) 時間複雜度
void sequentialScan(int *input, int *output, int n) {
    output[0] = input[0];
    for (int i = 1; i < n; i++) {
        output[i] = output[i-1] + input[i];
    }
}
```

## 平行掃描演算法

### 方法一：Naive 平行掃描

簡單但效率不高，O(n log n) 工作量。

```cuda
__global__ void naiveScan(int *input, int *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 複製輸入到輸出
    if (idx < n) {
        output[idx] = input[idx];
    }
    __syncthreads();

    // 迭代 log2(n) 次
    for (int stride = 1; stride < n; stride *= 2) {
        int val = 0;
        if (idx >= stride && idx < n) {
            val = output[idx - stride];
        }
        __syncthreads();

        if (idx >= stride && idx < n) {
            output[idx] += val;
        }
        __syncthreads();
    }
}
```

### 方法二：Blelloch Scan（Work-Efficient）

兩個階段：Up-Sweep 和 Down-Sweep，O(n) 工作量。

```
Up-Sweep（歸約）階段：
[3, 1, 7, 0, 4, 1, 6, 3]
[3, 4, 7, 7, 4, 5, 6, 9]
[3, 4, 7, 11, 4, 5, 6, 14]
[3, 4, 7, 11, 4, 5, 6, 25]

Down-Sweep 階段：
[3, 4, 7, 11, 4, 5, 6, 0]  ← 最後元素設為 0
[3, 4, 7, 0, 4, 5, 6, 11]
[3, 0, 7, 4, 4, 11, 6, 16]
[0, 3, 4, 11, 11, 15, 16, 22]  ← Exclusive Scan 結果
```

```cuda
#define BLOCK_SIZE 256

__global__ void blellochScan(int *data, int n) {
    __shared__ int temp[BLOCK_SIZE * 2];

    int thid = threadIdx.x;
    int offset = 1;

    // 載入資料到共享記憶體
    int ai = thid;
    int bi = thid + (n / 2);
    temp[ai] = data[ai];
    temp[bi] = data[bi];

    // Up-Sweep 階段
    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (thid < d) {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }

    // 清除最後一個元素
    if (thid == 0) {
        temp[n - 1] = 0;
    }

    // Down-Sweep 階段
    for (int d = 1; d < n; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (thid < d) {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;

            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }

    __syncthreads();

    // 寫回全域記憶體
    data[ai] = temp[ai];
    data[bi] = temp[bi];
}
```

## Block 級掃描（適用於大陣列）

對於超過一個 Block 的陣列，需要多階段處理：

```
階段 1：每個 Block 獨立掃描
Block 0: [3, 4, 11, 11] → sum = 11
Block 1: [4, 5, 11, 14] → sum = 14
Block 2: [6, 9, 15, 22] → sum = 22

階段 2：掃描 Block 總和
[11, 14, 22] → [0, 11, 25]

階段 3：將 Block 總和加到各 Block
Block 0: [3, 4, 11, 11] + 0
Block 1: [4, 5, 11, 14] + 11 = [15, 16, 22, 25]
Block 2: [6, 9, 15, 22] + 25 = [31, 34, 40, 47]
```

## 完整實作

```cuda
#include <stdio.h>

#define BLOCK_SIZE 256

// 單一 Block 的 Inclusive Scan
__device__ void warpScan(int *sdata, int lane) {
    if (lane >= 1)  sdata[lane] += sdata[lane - 1];
    if (lane >= 2)  sdata[lane] += sdata[lane - 2];
    if (lane >= 4)  sdata[lane] += sdata[lane - 4];
    if (lane >= 8)  sdata[lane] += sdata[lane - 8];
    if (lane >= 16) sdata[lane] += sdata[lane - 16];
}

__global__ void inclusiveScan(int *input, int *output, int *blockSums, int n) {
    __shared__ int sdata[BLOCK_SIZE];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // 載入資料
    sdata[tid] = (gid < n) ? input[gid] : 0;
    __syncthreads();

    // Warp 級掃描
    int lane = tid % 32;
    int warpId = tid / 32;

    warpScan(&sdata[warpId * 32], lane);
    __syncthreads();

    // 收集每個 Warp 的總和
    if (lane == 31) {
        sdata[BLOCK_SIZE + warpId] = sdata[tid];
    }
    __syncthreads();

    // 第一個 Warp 掃描 Warp 總和
    if (warpId == 0 && lane < (BLOCK_SIZE / 32)) {
        warpScan(&sdata[BLOCK_SIZE], lane);
    }
    __syncthreads();

    // 加上前面 Warp 的總和
    if (warpId > 0) {
        sdata[tid] += sdata[BLOCK_SIZE + warpId - 1];
    }
    __syncthreads();

    // 寫入結果
    if (gid < n) {
        output[gid] = sdata[tid];
    }

    // 儲存 Block 總和
    if (tid == BLOCK_SIZE - 1 && blockSums != NULL) {
        blockSums[blockIdx.x] = sdata[tid];
    }
}

// 將 Block 總和加到每個元素
__global__ void addBlockSums(int *output, int *blockSums, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid < n && blockIdx.x > 0) {
        output[gid] += blockSums[blockIdx.x - 1];
    }
}
```

## 應用：Stream Compaction

使用掃描實作資料壓縮（移除零元素）：

```cuda
// 步驟 1：創建標記陣列
// input:  [3, 0, 1, 0, 0, 2, 4, 0]
// flags:  [1, 0, 1, 0, 0, 1, 1, 0]

// 步驟 2：Exclusive Scan on flags
// scan:   [0, 1, 1, 2, 2, 2, 3, 4]

// 步驟 3：根據 scan 結果寫入
// output: [3, 1, 2, 4]

__global__ void scatter(int *input, int *flags, int *scanResult,
                        int *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n && flags[idx] == 1) {
        output[scanResult[idx]] = input[idx];
    }
}
```

## 效能比較

| 演算法 | 工作量 | 深度 | 優點 |
|--------|--------|------|------|
| Sequential | O(n) | O(n) | 簡單 |
| Naive Parallel | O(n log n) | O(log n) | 容易實作 |
| Blelloch | O(n) | O(log n) | 工作效率高 |

## 今日作業

1. 實作 Inclusive Scan
2. 實作 Exclusive Scan
3. 使用 Scan 實作 Stream Compaction
4. 比較不同實作的效能

## 🔧 編譯與執行

### CUDA 編譯

**Windows（cmd）：**

```cmd
nvcc -allow-unsupported-compiler -Wno-deprecated-gpu-targets -Xcompiler "/wd4819" -o scan_basic.exe scan_basic.cu
scan_basic.exe
```

**Windows（PowerShell）：**

```powershell
nvcc -allow-unsupported-compiler -Wno-deprecated-gpu-targets -Xcompiler "/wd4819" -o scan_basic.exe scan_basic.cu
.\scan_basic.exe
```

**WSL / Linux：**

```bash
nvcc -Wno-deprecated-gpu-targets -o scan_basic scan_basic.cu
./scan_basic
```

### Python 等效

```python
import cupy as cp
a = cp.array([1, 2, 3, 4, 5], dtype=cp.int32)
inclusive_scan = cp.cumsum(a)        # [1, 3, 6, 10, 15]
exclusive_scan = cp.concatenate([cp.array([0]), cp.cumsum(a)[:-1]])  # [0, 1, 3, 6, 10]
print(f"Inclusive: {inclusive_scan}")
print(f"Exclusive: {exclusive_scan}")
```

---

**明天我們將學習平行歸約（Reduction）的進階優化！**
