# Day 25: CUDA 動態平行處理（Dynamic Parallelism）

## 今日學習目標

- 理解動態平行處理的概念
- 學習在 GPU 核心中啟動核心
- 了解遞迴核心的應用
- 掌握動態平行處理的效能考量

## 什麼是動態平行處理？

傳統 CUDA：
```
CPU → 啟動核心 → GPU 執行
```

動態平行處理（Compute Capability 3.5+）：
```
CPU → 啟動核心 → GPU 核心可以啟動更多核心
```

## 為什麼需要動態平行處理？

### 適用場景

1. **遞迴演算法**：樹遍歷、分治法
2. **自適應演算法**：不規則資料結構
3. **減少 CPU-GPU 通訊**：避免來回傳輸控制資訊

### 傳統方式的問題

```
// 需要多次 CPU-GPU 交互
for (int level = 0; level < maxLevel; level++) {
    processLevel<<<...>>>(level);
    cudaDeviceSynchronize();  // 必須同步
    // 回到 CPU 決定下一步
}
```

### 動態平行處理的優勢

```cuda
// 全部在 GPU 上完成
__global__ void adaptiveProcess(...) {
    if (needMoreWork) {
        // 直接在 GPU 上啟動子核心
        adaptiveProcess<<<childBlocks, threads>>>(...);
    }
}
```

## 基本語法

### 在核心中啟動核心

```cuda
__global__ void parentKernel() {
    // 在 GPU 核心中啟動子核心
    childKernel<<<numBlocks, numThreads>>>(args);

    // 同步子核心
    cudaDeviceSynchronize();
}
```

### 完整範例

```cuda
#include <stdio.h>

__global__ void childKernel(int depth) {
    printf("    子核心: 深度 %d, 執行緒 %d\n", depth, threadIdx.x);
}

__global__ void parentKernel(int depth, int maxDepth) {
    printf("父核心: 深度 %d, 執行緒 %d\n", depth, threadIdx.x);

    if (depth < maxDepth) {
        // 每個執行緒啟動一個子核心
        childKernel<<<1, 2>>>(depth);
        cudaDeviceSynchronize();

        // 遞迴呼叫自己
        if (threadIdx.x == 0) {
            parentKernel<<<1, 2>>>(depth + 1, maxDepth);
            cudaDeviceSynchronize();
        }
    }
}

int main() {
    parentKernel<<<1, 2>>>(0, 3);
    cudaDeviceSynchronize();
    return 0;
}
```

## 遞迴核心：QuickSort 範例

```cuda
__device__ int partition(int *data, int left, int right) {
    int pivot = data[right];
    int i = left - 1;

    for (int j = left; j < right; j++) {
        if (data[j] <= pivot) {
            i++;
            // 交換
            int temp = data[i];
            data[i] = data[j];
            data[j] = temp;
        }
    }

    int temp = data[i + 1];
    data[i + 1] = data[right];
    data[right] = temp;

    return i + 1;
}

__global__ void quicksortKernel(int *data, int left, int right, int depth) {
    if (left >= right) return;

    // 當資料量小或深度太大時，用循序排序
    if (right - left < 32 || depth > 16) {
        // 簡單的插入排序
        for (int i = left + 1; i <= right; i++) {
            int key = data[i];
            int j = i - 1;
            while (j >= left && data[j] > key) {
                data[j + 1] = data[j];
                j--;
            }
            data[j + 1] = key;
        }
        return;
    }

    int pivotIdx = partition(data, left, right);

    // 在兩個 Streams 中平行處理左右子陣列
    cudaStream_t leftStream, rightStream;
    cudaStreamCreateWithFlags(&leftStream, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&rightStream, cudaStreamNonBlocking);

    // 啟動子核心
    if (pivotIdx - 1 > left) {
        quicksortKernel<<<1, 1, 0, leftStream>>>(data, left, pivotIdx - 1, depth + 1);
    }
    if (pivotIdx + 1 < right) {
        quicksortKernel<<<1, 1, 0, rightStream>>>(data, pivotIdx + 1, right, depth + 1);
    }

    cudaDeviceSynchronize();

    cudaStreamDestroy(leftStream);
    cudaStreamDestroy(rightStream);
}
```

## 自適應網格細化

```cuda
__global__ void adaptiveGrid(float *data, int level, int maxLevel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float value = data[idx];

    // 根據資料決定是否需要更細的處理
    if (needsRefinement(value) && level < maxLevel) {
        // 細化區域，啟動更多核心
        int childBlocks = 4;
        int childThreads = 32;
        adaptiveGrid<<<childBlocks, childThreads>>>(data, level + 1, maxLevel);
        cudaDeviceSynchronize();
    } else {
        // 直接處理
        processData(value);
    }
}
```

## 記憶體管理

### 裝置記憶體分配

```cuda
__global__ void kernelWithAlloc() {
    int *localData;

    // 在核心中分配記憶體
    localData = (int*)malloc(100 * sizeof(int));

    if (localData != NULL) {
        // 使用記憶體
        for (int i = 0; i < 100; i++) {
            localData[i] = i;
        }

        // 啟動子核心使用這個記憶體
        childKernel<<<1, 32>>>(localData);
        cudaDeviceSynchronize();

        // 釋放記憶體
        free(localData);
    }
}
```

### 設定 Device Heap 大小

```cuda
// 在主機端設定
size_t heapSize = 128 * 1024 * 1024;  // 128 MB
cudaDeviceSetLimit(cudaLimitMallocHeapSize, heapSize);
```

## 同步機制

### 隱式同步

```cuda
__global__ void parent() {
    child<<<1, 32>>>();
    // 核心結束時會隱式等待子核心
}
```

### 顯式同步

```cuda
__global__ void parent() {
    child<<<1, 32>>>();
    cudaDeviceSynchronize();  // 明確等待
    // 繼續處理
}
```

## 效能考量

### 開銷

- 核心啟動有開銷（~10 微秒）
- 不要啟動太多小核心
- 考慮使用 Streams 平行化

### 最佳實踐

```cuda
// 不好：太多小核心
for (int i = 0; i < 1000; i++) {
    tinyKernel<<<1, 1>>>();  // 開銷太大
}

// 好：合併工作
if (threadIdx.x < 1000) {
    processItem(threadIdx.x);
}
```

### 深度限制

```cuda
// 設定最大深度
if (depth < MAX_DEPTH) {
    childKernel<<<...>>>();
}
```

## 編譯選項

```bash
# 需要指定計算能力 3.5+
nvcc -arch=sm_35 -rdc=true dynamic.cu -o dynamic -lcudadevrt

# 對於較新的 GPU
nvcc -arch=sm_70 -rdc=true dynamic.cu -o dynamic -lcudadevrt
```

## 今日作業

1. 實作遞迴的 Fibonacci 核心
2. 使用動態平行處理實作樹遍歷
3. 比較動態平行與傳統方式的效能

## 🔧 編譯與執行

本日為動態平行處理概念說明，無範例程式需要編譯。

---

**明天我們將學習 Unified Memory 進階使用！**
