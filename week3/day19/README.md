# Day 19: 直方圖（Histogram）計算

## 今日學習目標

- 理解直方圖計算的平行挑戰
- 學習使用原子操作實作直方圖
- 探索共享記憶體優化技巧
- 了解私有化（Privatization）策略

## 什麼是直方圖？

直方圖統計資料中每個值出現的次數。

```
輸入資料: [1, 3, 2, 1, 3, 3, 2, 1, 4, 2]

直方圖（bins 0-4）:
bin 0: 0 次   ▓
bin 1: 3 次   ▓▓▓▓▓▓▓▓▓▓▓▓
bin 2: 3 次   ▓▓▓▓▓▓▓▓▓▓▓▓
bin 3: 3 次   ▓▓▓▓▓▓▓▓▓▓▓▓
bin 4: 1 次   ▓▓▓▓
```

## 平行化的挑戰

### 競爭條件（Race Condition）

多個執行緒同時更新同一個 bin：

```
Thread 1: 讀取 bin[3] = 5
Thread 2: 讀取 bin[3] = 5    ← 同時讀取
Thread 1: 寫入 bin[3] = 6
Thread 2: 寫入 bin[3] = 6    ← 覆蓋！應該是 7
```

解決方案：**原子操作**

## 實作方法

### 方法一：全域記憶體原子操作（簡單但慢）

```cuda
__global__ void histogramGlobal(unsigned char *data, int *histogram, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        int bin = data[idx];
        atomicAdd(&histogram[bin], 1);  // 原子加法
    }
}
```

**問題**：大量競爭造成效能瓶頸

### 方法二：共享記憶體私有化

每個 Block 使用私有直方圖，最後合併：

```cuda
#define NUM_BINS 256

__global__ void histogramShared(unsigned char *data, int *histogram, int n) {
    __shared__ int localHist[NUM_BINS];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 初始化共享記憶體
    if (tid < NUM_BINS) {
        localHist[tid] = 0;
    }
    __syncthreads();

    // 累積到局部直方圖
    if (idx < n) {
        int bin = data[idx];
        atomicAdd(&localHist[bin], 1);
    }
    __syncthreads();

    // 合併到全域直方圖
    if (tid < NUM_BINS) {
        atomicAdd(&histogram[tid], localHist[tid]);
    }
}
```

### 方法三：執行緒私有化（更激進）

每個執行緒有私有計數器，減少原子操作：

```cuda
#define NUM_BINS 256
#define THREADS_PER_BLOCK 256
#define ITEMS_PER_THREAD 16

__global__ void histogramPrivate(unsigned char *data, int *histogram, int n) {
    __shared__ int localHist[NUM_BINS];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x * ITEMS_PER_THREAD + tid;

    // 執行緒私有計數器
    int privateHist[NUM_BINS];
    for (int i = 0; i < NUM_BINS; i++) {
        privateHist[i] = 0;
    }

    // 每個執行緒處理多個元素
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int idx = gid + i * THREADS_PER_BLOCK;
        if (idx < n) {
            int bin = data[idx];
            privateHist[bin]++;  // 無競爭！
        }
    }

    // 初始化共享直方圖
    if (tid < NUM_BINS) {
        localHist[tid] = 0;
    }
    __syncthreads();

    // 合併私有到共享
    for (int i = 0; i < NUM_BINS; i++) {
        if (privateHist[i] > 0) {
            atomicAdd(&localHist[i], privateHist[i]);
        }
    }
    __syncthreads();

    // 合併共享到全域
    if (tid < NUM_BINS) {
        atomicAdd(&histogram[tid], localHist[tid]);
    }
}
```

## 優化技巧

### 1. 減少 Bank Conflicts

在共享記憶體中加入 padding：

```cuda
// 可能有 Bank Conflict
__shared__ int localHist[256];

// 減少 Bank Conflict
__shared__ int localHist[256 + 1];  // 加 padding
```

### 2. 交錯存取模式

讓連續執行緒存取不同的 bin：

```cuda
// 處理連續資料時，執行緒會競爭相似的 bin
int bin = data[idx];

// 可以考慮打亂執行緒的處理順序
int shuffledIdx = /* 某種打亂邏輯 */;
int bin = data[shuffledIdx];
```

### 3. Warp 級聚合

同一 Warp 內的執行緒先聚合：

```cuda
__device__ void warpHistogram(int bin, int *hist) {
    unsigned int mask = __ballot_sync(0xffffffff, 1);

    // 找出 Warp 中有相同 bin 的執行緒
    for (int i = 0; i < 32; i++) {
        unsigned int peers = __ballot_sync(mask, bin == i);
        if (bin == i) {
            int leader = __ffs(peers) - 1;
            int count = __popc(peers);
            if (threadIdx.x % 32 == leader) {
                atomicAdd(&hist[i], count);
            }
        }
    }
}
```

## 完整範例程式

```cuda
#include <stdio.h>
#include <stdlib.h>

#define NUM_BINS 256
#define BLOCK_SIZE 256

// 方法 1：全域原子操作
__global__ void histogramV1(unsigned char *data, unsigned int *hist, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicAdd(&hist[data[idx]], 1);
    }
}

// 方法 2：共享記憶體
__global__ void histogramV2(unsigned char *data, unsigned int *hist, int n) {
    __shared__ unsigned int localHist[NUM_BINS];

    // 初始化
    for (int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
        localHist[i] = 0;
    }
    __syncthreads();

    // 累積
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicAdd(&localHist[data[idx]], 1);
    }
    __syncthreads();

    // 合併
    for (int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
        atomicAdd(&hist[i], localHist[i]);
    }
}

int main() {
    int n = 1 << 20;  // 1M 元素

    // 分配記憶體
    unsigned char *h_data = (unsigned char*)malloc(n);
    unsigned int *h_hist = (unsigned int*)calloc(NUM_BINS, sizeof(unsigned int));

    // 初始化隨機資料
    for (int i = 0; i < n; i++) {
        h_data[i] = rand() % NUM_BINS;
    }

    // GPU 記憶體
    unsigned char *d_data;
    unsigned int *d_hist;
    cudaMalloc(&d_data, n);
    cudaMalloc(&d_hist, NUM_BINS * sizeof(unsigned int));

    cudaMemcpy(d_data, h_data, n, cudaMemcpyHostToDevice);
    cudaMemset(d_hist, 0, NUM_BINS * sizeof(unsigned int));

    // 執行
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    histogramV2<<<blocks, BLOCK_SIZE>>>(d_data, d_hist, n);

    // 取回結果
    cudaMemcpy(h_hist, d_hist, NUM_BINS * sizeof(unsigned int),
               cudaMemcpyDeviceToHost);

    // 顯示部分結果
    printf("直方圖前 10 個 bins:\n");
    for (int i = 0; i < 10; i++) {
        printf("bin[%d] = %u\n", i, h_hist[i]);
    }

    // 清理
    free(h_data);
    free(h_hist);
    cudaFree(d_data);
    cudaFree(d_hist);

    return 0;
}
```

## 應用場景

1. **圖像處理**：計算像素分布
2. **資料分析**：統計資料分布
3. **機器學習**：特徵工程
4. **科學計算**：蒙地卡羅模擬

## 效能比較

| 方法 | 優點 | 缺點 |
|------|------|------|
| 全域原子 | 簡單 | 競爭嚴重 |
| 共享記憶體 | 減少競爭 | 需要額外同步 |
| 執行緒私有 | 最少競爭 | 記憶體用量大 |

## 今日作業

1. 實作三種直方圖方法並比較效能
2. 對圖像資料計算 RGB 直方圖
3. 嘗試不同的 Block 大小觀察效能變化

## 🔧 編譯與執行

### CUDA 編譯

**Windows（cmd）：**

```cmd
nvcc -allow-unsupported-compiler -Wno-deprecated-gpu-targets -Xcompiler "/wd4819" -o histogram.exe histogram.cu
histogram.exe
```

**Windows（PowerShell）：**

```powershell
nvcc -allow-unsupported-compiler -Wno-deprecated-gpu-targets -Xcompiler "/wd4819" -o histogram.exe histogram.cu
.\histogram.exe
```

**WSL / Linux：**

```bash
nvcc -Wno-deprecated-gpu-targets -o histogram histogram.cu
./histogram
```

### Python 等效

```python
import cupy as cp
data = cp.random.randint(0, 256, size=100000, dtype=cp.int32)
hist = cp.bincount(data, minlength=256)
print(f"直方圖前 10 個 bin: {hist[:10]}")
```

---

**明天我們將學習排序演算法的 GPU 實作！**
