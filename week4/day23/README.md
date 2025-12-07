# Day 23: CUDA Streams 與非同步執行

## 今日學習目標

- 理解 CUDA Streams 的概念
- 學習如何重疊計算和資料傳輸
- 實作多 Stream 平行處理
- 掌握非同步 API 的使用

## 什麼是 CUDA Stream？

Stream 是一個在 GPU 上按順序執行的命令佇列。

```
預設 Stream（Stream 0）：所有操作依序執行

Stream 1: [Kernel A] → [Kernel B]
Stream 2: [Kernel C] → [Kernel D]  ← 可以與 Stream 1 平行

多個 Streams 可以同時執行！
```

## 為什麼使用 Streams？

### 問題：傳統執行模式

```
時間線：
[H2D 傳輸] → [核心執行] → [D2H 傳輸]
     ↑            ↑            ↑
    等待         等待         等待
```

### 解決方案：Stream 重疊

```
Stream 0: [H2D_0] → [Kernel_0] → [D2H_0]
Stream 1:      [H2D_1] → [Kernel_1] → [D2H_1]
Stream 2:           [H2D_2] → [Kernel_2] → [D2H_2]
          └──────────────────────────────────────→ 時間

節省時間！
```

## 基本 API

### 創建和銷毀 Stream

```cuda
cudaStream_t stream;
cudaStreamCreate(&stream);     // 創建
// ... 使用 stream ...
cudaStreamDestroy(stream);     // 銷毀
```

### 非同步記憶體傳輸

```cuda
// 需要使用 Pinned Memory
cudaMallocHost(&h_data, size);  // 分配 pinned memory

// 非同步傳輸（不會阻塞 CPU）
cudaMemcpyAsync(d_data, h_data, size,
                cudaMemcpyHostToDevice, stream);
```

### 非同步核心啟動

```cuda
// 核心在指定 stream 中執行
myKernel<<<blocks, threads, 0, stream>>>(args);
```

## Pinned Memory（頁鎖定記憶體）

### 為什麼需要 Pinned Memory？

```
普通記憶體（Pageable）:
  CPU Memory → 緩衝區 → GPU Memory
                 ↑
            需要額外複製

Pinned Memory（Page-locked）:
  CPU Memory ────────→ GPU Memory
                 ↑
            直接 DMA 傳輸
```

### 使用方式

```cuda
float *h_pinned;
cudaMallocHost(&h_pinned, size);  // 分配 pinned memory
// 或
cudaHostAlloc(&h_pinned, size, cudaHostAllocDefault);

// 使用完畢後
cudaFreeHost(h_pinned);
```

## 完整範例：重疊傳輸與計算

```cuda
#include <stdio.h>

#define N (1 << 20)
#define NSTREAMS 4

__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    int size = N * sizeof(float);
    int streamSize = N / NSTREAMS;
    int streamBytes = streamSize * sizeof(float);

    // 分配 Pinned Memory
    float *h_a, *h_b, *h_c;
    cudaMallocHost(&h_a, size);
    cudaMallocHost(&h_b, size);
    cudaMallocHost(&h_c, size);

    // 初始化資料
    for (int i = 0; i < N; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    // 分配 GPU 記憶體
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // 創建 Streams
    cudaStream_t streams[NSTREAMS];
    for (int i = 0; i < NSTREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // 使用多個 Streams
    for (int i = 0; i < NSTREAMS; i++) {
        int offset = i * streamSize;

        // 非同步複製 H2D
        cudaMemcpyAsync(&d_a[offset], &h_a[offset], streamBytes,
                        cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(&d_b[offset], &h_b[offset], streamBytes,
                        cudaMemcpyHostToDevice, streams[i]);

        // 核心執行
        int threads = 256;
        int blocks = (streamSize + threads - 1) / threads;
        vectorAdd<<<blocks, threads, 0, streams[i]>>>
            (&d_a[offset], &d_b[offset], &d_c[offset], streamSize);

        // 非同步複製 D2H
        cudaMemcpyAsync(&h_c[offset], &d_c[offset], streamBytes,
                        cudaMemcpyDeviceToHost, streams[i]);
    }

    // 同步所有 Streams
    for (int i = 0; i < NSTREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    // 驗證結果
    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (h_c[i] != 3.0f) {
            correct = false;
            break;
        }
    }
    printf("結果: %s\n", correct ? "正確" : "錯誤");

    // 清理
    for (int i = 0; i < NSTREAMS; i++) {
        cudaStreamDestroy(streams[i]);
    }

    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
```

## Stream 同步

### 方法一：同步單一 Stream

```cuda
cudaStreamSynchronize(stream);  // 等待該 stream 完成
```

### 方法二：同步所有 Streams

```cuda
cudaDeviceSynchronize();  // 等待所有 streams 完成
```

### 方法三：使用事件（Events）

```cuda
cudaEvent_t event;
cudaEventCreate(&event);

cudaEventRecord(event, stream);  // 在 stream 中記錄事件
cudaEventSynchronize(event);      // 等待事件完成

cudaEventDestroy(event);
```

## Stream 間的依賴

### 使用事件建立依賴

```cuda
cudaEvent_t event;
cudaEventCreate(&event);

// Stream 1 中的操作
kernel1<<<..., 0, stream1>>>();
cudaEventRecord(event, stream1);

// Stream 2 等待 Stream 1 的事件
cudaStreamWaitEvent(stream2, event, 0);
kernel2<<<..., 0, stream2>>>();
```

## 效能分析：計時

```cuda
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start, stream);
// ... 操作 ...
cudaEventRecord(stop, stream);

cudaEventSynchronize(stop);

float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);
printf("執行時間: %.3f ms\n", milliseconds);
```

## Stream 優先級

```cuda
// 獲取優先級範圍
int leastPriority, greatestPriority;
cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);

// 創建高優先級 Stream
cudaStream_t highPriorityStream;
cudaStreamCreateWithPriority(&highPriorityStream,
                              cudaStreamNonBlocking,
                              greatestPriority);
```

## 最佳實踐

1. **使用 Pinned Memory**：非同步傳輸需要頁鎖定記憶體
2. **適當的 Stream 數量**：通常 2-4 個效果最好
3. **避免假依賴**：使用 `cudaStreamNonBlocking`
4. **交錯操作**：將傳輸和計算交織在一起

## 今日作業

1. 實作多 Stream 向量加法
2. 比較單 Stream 和多 Stream 的效能
3. 使用事件測量各階段時間

## 編譯與執行

```bash
nvcc streams.cu -o streams
./streams
```

---

**明天我們將學習多 GPU 程式設計！**
