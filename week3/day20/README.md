# Day 20: 排序演算法 - Bitonic Sort 與 Radix Sort

## 今日學習目標

- 理解 GPU 排序的特殊挑戰
- 學習 Bitonic Sort（雙調排序）
- 了解 Radix Sort（基數排序）原理
- 比較不同排序演算法的效能

## GPU 排序的挑戰

傳統排序演算法（Quick Sort、Merge Sort）在 GPU 上效率不高：

- **資料依賴**：需要比較和交換
- **不規則存取**：快速排序的分區操作
- **遞迴結構**：GPU 不擅長遞迴

## Bitonic Sort（雙調排序）

### 什麼是 Bitonic 序列？

先遞增後遞減（或反過來）的序列：

```
Bitonic 序列範例:
[1, 3, 5, 7, 6, 4, 2, 0]  ← 先升後降
[7, 5, 3, 1, 2, 4, 6, 8]  ← 先降後升
```

### Bitonic Sort 原理

1. 將序列構造成 Bitonic 序列
2. 使用 Bitonic Merge 排序

### 視覺化過程

```
初始:     [3, 7, 4, 8, 6, 2, 1, 5]

構造 Bitonic 序列:
Step 1:   比較距離 1 的元素
          [3,7] [8,4] [2,6] [5,1]
          ↓     ↓     ↓     ↓
          [3,7] [4,8] [2,6] [1,5]

Step 2:   比較距離 2 的元素，然後距離 1
          [3,7,4,8] ↔ [2,6,1,5]
          ...

最終排序: [1, 2, 3, 4, 5, 6, 7, 8]
```

### CUDA 實作

```cuda
#include <stdio.h>

#define BLOCK_SIZE 256

__global__ void bitonicSort(int *data, int j, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int ixj = idx ^ j;  // 找到配對的元素

    if (ixj > idx) {
        if ((idx & k) == 0) {
            // 升序比較
            if (data[idx] > data[ixj]) {
                // 交換
                int temp = data[idx];
                data[idx] = data[ixj];
                data[ixj] = temp;
            }
        } else {
            // 降序比較
            if (data[idx] < data[ixj]) {
                // 交換
                int temp = data[idx];
                data[idx] = data[ixj];
                data[ixj] = temp;
            }
        }
    }
}

void bitonicSortHost(int *d_data, int n) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // k 是 Bitonic 序列的大小
    for (int k = 2; k <= n; k *= 2) {
        // j 是比較的距離
        for (int j = k / 2; j > 0; j /= 2) {
            bitonicSort<<<blocks, BLOCK_SIZE>>>(d_data, j, k);
            cudaDeviceSynchronize();
        }
    }
}
```

### 共享記憶體優化版本

```cuda
__global__ void bitonicSortShared(int *data, int n) {
    __shared__ int sdata[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // 載入到共享記憶體
    sdata[tid] = (idx < n) ? data[idx] : INT_MAX;
    __syncthreads();

    // 在共享記憶體中排序
    for (int k = 2; k <= BLOCK_SIZE; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            int ixj = tid ^ j;
            if (ixj > tid) {
                if ((tid & k) == 0) {
                    if (sdata[tid] > sdata[ixj]) {
                        int temp = sdata[tid];
                        sdata[tid] = sdata[ixj];
                        sdata[ixj] = temp;
                    }
                } else {
                    if (sdata[tid] < sdata[ixj]) {
                        int temp = sdata[tid];
                        sdata[tid] = sdata[ixj];
                        sdata[ixj] = temp;
                    }
                }
            }
            __syncthreads();
        }
    }

    // 寫回全域記憶體
    if (idx < n) {
        data[idx] = sdata[tid];
    }
}
```

## Radix Sort（基數排序）

### 原理

按照數字的每一位進行排序，從最低位到最高位：

```
初始:  [329, 457, 657, 839, 436, 720, 355]

按個位: [720, 355, 436, 457, 657, 329, 839]
按十位: [720, 329, 436, 839, 355, 457, 657]
按百位: [329, 355, 436, 457, 657, 720, 839]

結果:  [329, 355, 436, 457, 657, 720, 839]
```

### GPU Radix Sort 步驟

1. **計算直方圖**：統計每個位的值分布
2. **Exclusive Scan**：計算每個值的偏移
3. **重新排列**：根據偏移移動元素

### 簡化實作（4-bit Radix Sort）

```cuda
#define RADIX_BITS 4
#define RADIX_SIZE (1 << RADIX_BITS)  // 16

__global__ void countRadix(unsigned int *data, unsigned int *counts,
                           int n, int shift) {
    __shared__ unsigned int localCounts[RADIX_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // 初始化
    if (tid < RADIX_SIZE) {
        localCounts[tid] = 0;
    }
    __syncthreads();

    // 計數
    if (idx < n) {
        unsigned int radix = (data[idx] >> shift) & (RADIX_SIZE - 1);
        atomicAdd(&localCounts[radix], 1);
    }
    __syncthreads();

    // 合併到全域
    if (tid < RADIX_SIZE) {
        atomicAdd(&counts[tid], localCounts[tid]);
    }
}

__global__ void reorder(unsigned int *input, unsigned int *output,
                        unsigned int *offsets, int n, int shift) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        unsigned int value = input[idx];
        unsigned int radix = (value >> shift) & (RADIX_SIZE - 1);

        // 使用原子操作獲取輸出位置
        unsigned int pos = atomicAdd(&offsets[radix], 1);
        output[pos] = value;
    }
}
```

## 效能比較

| 演算法 | 時間複雜度 | 空間 | GPU 適合度 |
|--------|------------|------|------------|
| Quick Sort | O(n log n) | O(log n) | 低 |
| Merge Sort | O(n log n) | O(n) | 中 |
| Bitonic Sort | O(n log²n) | O(1) | 高 |
| Radix Sort | O(n·k) | O(n) | 高 |

## 使用 Thrust 函式庫

對於實際應用，建議使用 Thrust：

```cuda
#include <thrust/device_vector.h>
#include <thrust/sort.h>

int main() {
    // 初始化資料
    thrust::device_vector<int> d_vec(1000000);

    // 填充隨機資料
    thrust::generate(d_vec.begin(), d_vec.end(), rand);

    // 排序（自動選擇最佳演算法）
    thrust::sort(d_vec.begin(), d_vec.end());

    return 0;
}
```

### Thrust 排序特點

- 自動選擇最佳演算法
- 支援自訂比較函數
- 穩定排序選項
- 高度優化

## 今日作業

1. 實作 Bitonic Sort
2. 使用 Thrust 排序並比較效能
3. 嘗試對結構體排序

## 🔧 編譯與執行

### CUDA 編譯

**Windows（cmd）：**

```cmd
nvcc -allow-unsupported-compiler -Wno-deprecated-gpu-targets -Xcompiler "/wd4819" -o bitonic_sort.exe bitonic_sort.cu
bitonic_sort.exe
```

**Windows（PowerShell）：**

```powershell
nvcc -allow-unsupported-compiler -Wno-deprecated-gpu-targets -Xcompiler "/wd4819" -o bitonic_sort.exe bitonic_sort.cu
.\bitonic_sort.exe
```

**WSL / Linux：**

```bash
nvcc -Wno-deprecated-gpu-targets -o bitonic_sort bitonic_sort.cu
./bitonic_sort
```

### Python 等效

```python
import cupy as cp
data = cp.random.rand(1024, dtype=cp.float32)
sorted_data = cp.sort(data)  # GPU 排序
print(f"前 10 個: {sorted_data[:10]}")
```

---

**明天是週末專題：完整的圖像處理應用！**
