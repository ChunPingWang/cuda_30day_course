# Day 8: Grid、Block 和 Thread 的層次結構

## 📚 今日學習目標

- 深入理解 CUDA 執行模型
- 學習 Warp 的概念
- 理解執行緒的硬體對應
- 學習如何選擇最佳的 Block 大小

## 🏗️ CUDA 執行模型深入

### 軟體層次

```
Grid（網格）
 │
 ├── Block 0
 │    ├── Thread 0, 1, 2, ..., N-1
 │
 ├── Block 1
 │    ├── Thread 0, 1, 2, ..., N-1
 │
 └── Block 2
      ├── Thread 0, 1, 2, ..., N-1
```

### 硬體對應

```
GPU 晶片
 │
 ├── SM 0 (Streaming Multiprocessor)
 │    ├── 執行 Block X
 │    └── 執行 Block Y（如果資源足夠）
 │
 ├── SM 1
 │    ├── 執行 Block Z
 │    └── ...
 │
 └── SM N
      └── ...
```

## 🔄 什麼是 Warp？

**Warp** 是 CUDA 執行的基本單位！

### 重要概念

- 一個 Warp = 32 個連續的執行緒
- 同一個 Warp 中的執行緒會同時執行相同的指令
- 這稱為 **SIMT**（Single Instruction, Multiple Threads）

### 範例

如果 Block 有 128 個執行緒：
```
Block (128 threads)
 │
 ├── Warp 0: Thread 0-31
 ├── Warp 1: Thread 32-63
 ├── Warp 2: Thread 64-95
 └── Warp 3: Thread 96-127
```

## ⚠️ Warp Divergence（分歧）

當同一個 Warp 中的執行緒走不同的分支，效能會下降！

### 問題範例

```cuda
__global__ void badKernel(int *data) {
    int idx = threadIdx.x;

    if (idx % 2 == 0) {
        // 偶數執行緒走這裡
        doSomething();
    } else {
        // 奇數執行緒走這裡
        doSomethingElse();
    }
}
```

### 發生什麼事？

```
Warp 0 (Thread 0-31):
  Step 1: Thread 0, 2, 4, ... 執行 if 分支（其他等待）
  Step 2: Thread 1, 3, 5, ... 執行 else 分支（其他等待）
```

效能降低約 50%！

### 好的做法

```cuda
__global__ void goodKernel(int *data) {
    int idx = threadIdx.x;
    int warpId = idx / 32;

    if (warpId % 2 == 0) {
        // 整個 Warp 0, 2, 4, ... 走這裡
        doSomething();
    } else {
        // 整個 Warp 1, 3, 5, ... 走這裡
        doSomethingElse();
    }
}
```

## 📊 Block 大小選擇

### 規則

1. **必須是 32 的倍數**（Warp 大小）
2. **常用選擇**：128、256、512、1024
3. **最大限制**：每個 Block 最多 1024 個執行緒

### 考慮因素

| Block 大小 | 優點 | 缺點 |
|-----------|------|------|
| 較小(64-128) | 更多 Block 可並行 | 共享記憶體使用效率低 |
| 較大(512-1024) | 更好的資源共享 | Block 數量少，SM 使用率低 |

### 推薦

```cuda
// 通常的選擇
int threadsPerBlock = 256;
int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
```

## 🔢 GPU 規格查詢

你的 RTX 4060 Laptop 規格：
- **SM 數量**: 24
- **每個 SM 最大執行緒數**: 1536
- **每個 Block 最大執行緒數**: 1024
- **Warp 大小**: 32
- **最大 Block 維度**: (1024, 1024, 64)
- **最大 Grid 維度**: (2³¹-1, 65535, 65535)

## 💡 佔用率（Occupancy）

**佔用率** = 實際執行的 Warp 數 / SM 最大 Warp 數

### 影響因素

1. **Block 大小**
2. **暫存器使用量**
3. **共享記憶體使用量**

### 查詢佔用率

```cuda
int blockSize = 256;
int minGridSize, gridSize;

// 讓 CUDA 建議最佳配置
cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, myKernel, 0, 0);
```

## 🔧 實作練習

### 範例程式

1. **warp_info.cu** - 展示 Warp 資訊
2. **occupancy.cu** - 佔用率計算
3. **divergence_demo.cu** - Warp 分歧示範

## 📝 今日作業

1. ✅ 理解 Warp 的概念
2. ✅ 執行 `warp_info.cu`
3. ✅ 理解 Warp Divergence
4. ✅ 完成練習題

## 🎯 練習題

### 練習 1
一個核心函數使用 `<<<10, 192>>>` 配置。
- 有多少個 Warp？
- 最後一個 Block 有沒有不完整的 Warp？

### 練習 2
修改一段有 Warp Divergence 的程式碼，減少分歧。

## 🤓 重要公式

### Warp 數計算
```
Block 中的 Warp 數 = ceil(threadsPerBlock / 32)
```

### 佔用率
```
佔用率 = (活躍 Warp / SM 最大 Warp) × 100%
```

## ❓ 思考問題

1. 為什麼 Warp 大小是 32？
2. 如果一個 Block 有 100 個執行緒，會有幾個 Warp？最後一個 Warp 的情況？
3. 為什麼高佔用率不一定意味著高效能？

---

**明天我們將學習向量運算的最佳化！** ⚡
