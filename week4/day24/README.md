# Day 24: 多 GPU 程式設計

## 今日學習目標

- 了解多 GPU 系統架構
- 學習如何在多個 GPU 間分配工作
- 掌握 GPU 間資料傳輸技術
- 實作多 GPU 平行運算

## 多 GPU 系統概述

### 常見架構

```
┌─────────────────────────────────────┐
│              CPU                    │
│          主記憶體                   │
└──────────┬───────────┬──────────────┘
           │           │
      ┌────▼────┐ ┌────▼────┐
      │  GPU 0  │ │  GPU 1  │
      │ 記憶體  │ │ 記憶體  │
      └─────────┘ └─────────┘
```

### NVLink 連接（高速 GPU 互連）

```
┌─────────┐     NVLink     ┌─────────┐
│  GPU 0  │ ←───────────→ │  GPU 1  │
└─────────┘    高速互連     └─────────┘
```

## 基本 API

### 查詢 GPU 數量

```cuda
int deviceCount;
cudaGetDeviceCount(&deviceCount);
printf("系統中有 %d 個 GPU\n", deviceCount);
```

### 選擇 GPU

```cuda
// 設定當前使用的 GPU
cudaSetDevice(0);  // 使用 GPU 0
// ... 在 GPU 0 上的操作 ...

cudaSetDevice(1);  // 切換到 GPU 1
// ... 在 GPU 1 上的操作 ...
```

### 查詢 GPU 資訊

```cuda
for (int i = 0; i < deviceCount; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("GPU %d: %s\n", i, prop.name);
    printf("  記憶體: %.1f GB\n", prop.totalGlobalMem / 1e9);
}
```

## 多 GPU 記憶體管理

### 各 GPU 獨立分配

```cuda
float *d_data0, *d_data1;

// 在 GPU 0 分配
cudaSetDevice(0);
cudaMalloc(&d_data0, size);

// 在 GPU 1 分配
cudaSetDevice(1);
cudaMalloc(&d_data1, size);
```

### GPU 間資料傳輸

```cuda
// 方法一：通過 CPU
cudaMemcpy(h_temp, d_data0, size, cudaMemcpyDeviceToHost);
cudaMemcpy(d_data1, h_temp, size, cudaMemcpyHostToDevice);

// 方法二：Peer-to-Peer（如果支援）
cudaMemcpyPeer(d_data1, 1, d_data0, 0, size);
```

## Peer-to-Peer 存取

### 檢查支援

```cuda
int canAccessPeer;
cudaDeviceCanAccessPeer(&canAccessPeer, 0, 1);  // GPU 0 能否存取 GPU 1
if (canAccessPeer) {
    printf("支援 P2P 存取\n");
}
```

### 啟用 P2P

```cuda
// GPU 0 可以直接存取 GPU 1 的記憶體
cudaSetDevice(0);
cudaDeviceEnablePeerAccess(1, 0);

// GPU 1 可以直接存取 GPU 0 的記憶體
cudaSetDevice(1);
cudaDeviceEnablePeerAccess(0, 0);
```

### P2P 直接存取

```cuda
// 在 GPU 0 的核心中直接存取 GPU 1 的資料
__global__ void kernelWithP2P(float *localData, float *remoteData, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 直接讀取另一個 GPU 的資料
        localData[idx] = remoteData[idx] * 2.0f;
    }
}
```

## 工作分割策略

### 策略一：資料平行（最常用）

```cuda
// 將資料分成 N 等份，每個 GPU 處理一份
int chunkSize = totalSize / numGPUs;

for (int i = 0; i < numGPUs; i++) {
    cudaSetDevice(i);
    int offset = i * chunkSize;
    processData<<<blocks, threads>>>(&d_data[i][0], chunkSize);
}
```

### 策略二：管線平行

```cuda
// 不同 GPU 執行不同階段
// GPU 0: 階段 1
// GPU 1: 階段 2
// GPU 2: 階段 3
```

### 策略三：模型平行

```cuda
// 適用於深度學習
// 將大模型分散到多個 GPU
```

## 完整範例：多 GPU 向量加法

```cuda
#include <stdio.h>

#define N (1 << 24)  // 16M 元素

__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("找到 %d 個 GPU\n", deviceCount);

    if (deviceCount < 2) {
        printf("需要至少 2 個 GPU\n");
        return 1;
    }

    int size = N * sizeof(float);
    int halfSize = size / 2;
    int halfN = N / 2;

    // 主機記憶體
    float *h_a, *h_b, *h_c;
    cudaMallocHost(&h_a, size);
    cudaMallocHost(&h_b, size);
    cudaMallocHost(&h_c, size);

    // 初始化
    for (int i = 0; i < N; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    // 各 GPU 的記憶體和 Streams
    float *d_a[2], *d_b[2], *d_c[2];
    cudaStream_t streams[2];

    for (int i = 0; i < 2; i++) {
        cudaSetDevice(i);
        cudaMalloc(&d_a[i], halfSize);
        cudaMalloc(&d_b[i], halfSize);
        cudaMalloc(&d_c[i], halfSize);
        cudaStreamCreate(&streams[i]);
    }

    // 計時
    cudaEvent_t start, stop;
    cudaSetDevice(0);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // 在兩個 GPU 上平行執行
    for (int i = 0; i < 2; i++) {
        cudaSetDevice(i);
        int offset = i * halfN;

        // 非同步傳輸
        cudaMemcpyAsync(d_a[i], h_a + offset, halfSize,
                        cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(d_b[i], h_b + offset, halfSize,
                        cudaMemcpyHostToDevice, streams[i]);

        // 計算
        int threads = 256;
        int blocks = (halfN + threads - 1) / threads;
        vectorAdd<<<blocks, threads, 0, streams[i]>>>(d_a[i], d_b[i], d_c[i], halfN);

        // 結果傳回
        cudaMemcpyAsync(h_c + offset, d_c[i], halfSize,
                        cudaMemcpyDeviceToHost, streams[i]);
    }

    // 同步所有 GPU
    for (int i = 0; i < 2; i++) {
        cudaSetDevice(i);
        cudaStreamSynchronize(streams[i]);
    }

    cudaSetDevice(0);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // 驗證
    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (h_c[i] != 3.0f) {
            correct = false;
            break;
        }
    }

    printf("多 GPU 執行時間: %.3f ms\n", milliseconds);
    printf("結果驗證: %s\n", correct ? "正確" : "錯誤");

    // 清理
    for (int i = 0; i < 2; i++) {
        cudaSetDevice(i);
        cudaFree(d_a[i]);
        cudaFree(d_b[i]);
        cudaFree(d_c[i]);
        cudaStreamDestroy(streams[i]);
    }

    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);

    return 0;
}
```

## 多 GPU 同步

### 使用事件同步

```cuda
cudaEvent_t events[2];

for (int i = 0; i < 2; i++) {
    cudaSetDevice(i);
    cudaEventCreate(&events[i]);
    kernel<<<...>>>(...);
    cudaEventRecord(events[i]);
}

// 等待所有 GPU 完成
for (int i = 0; i < 2; i++) {
    cudaEventSynchronize(events[i]);
}
```

## 效能考量

1. **通訊開銷**：減少 GPU 間的資料傳輸
2. **負載平衡**：確保工作量均勻分配
3. **P2P 效能**：NVLink 比 PCIe 快很多
4. **記憶體限制**：每個 GPU 有獨立的記憶體

## 今日作業

1. 查詢你系統中的 GPU 數量和規格
2. 如果有多個 GPU，實作多 GPU 向量加法
3. 比較單 GPU 和多 GPU 的效能

## 編譯與執行

```bash
nvcc multi_gpu.cu -o multi_gpu
./multi_gpu
```

---

**明天我們將學習 CUDA 動態平行處理！**
