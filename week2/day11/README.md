# Day 11: 共享記憶體（Shared Memory）

## 📚 今日學習目標

- 理解共享記憶體的特性
- 學習如何宣告和使用共享記憶體
- 使用共享記憶體優化矩陣乘法
- 理解 Bank Conflict（下一課會深入）

## 💾 什麼是共享記憶體？

共享記憶體是一塊**晶片上的高速記憶體**，每個 Block 都有自己獨立的共享記憶體。

### 特點

| 特性 | 說明 |
|------|------|
| 位置 | GPU 晶片上（非顯存） |
| 速度 | 比全域記憶體快 ~100 倍 |
| 大小 | 每個 Block 約 48-164 KB |
| 作用域 | 同一個 Block 內的執行緒共享 |
| 生命週期 | 核心函數執行期間 |

### 記憶體速度比較

```
暫存器   ████████████████████████████████████████  最快
共享記憶體 ███████████████████████████████         很快
L1 Cache ██████████████████████                   快
L2 Cache ██████████████                           中等
全域記憶體 ███                                     最慢
```

## 🔧 宣告共享記憶體

### 靜態分配（編譯時確定大小）

```cuda
__global__ void myKernel() {
    // 宣告一個 256 個 float 的共享記憶體陣列
    __shared__ float sharedData[256];

    // 使用共享記憶體
    sharedData[threadIdx.x] = ...;
}
```

### 動態分配（執行時確定大小）

```cuda
// 核心函數宣告使用 extern
__global__ void myKernel() {
    extern __shared__ float sharedData[];
    // ...
}

// 啟動時指定大小
myKernel<<<blocks, threads, sharedMemBytes>>>();
```

## 🔄 共享記憶體的典型用法

### 模式 1：資料重用

```cuda
__global__ void example(float *input, float *output, int n) {
    __shared__ float tile[BLOCK_SIZE];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 1. 從全域記憶體載入到共享記憶體
    if (idx < n) {
        tile[threadIdx.x] = input[idx];
    }

    // 2. 同步：確保所有執行緒都載入完成
    __syncthreads();

    // 3. 使用共享記憶體中的資料（可重複使用！）
    // ...

    // 4. 寫回全域記憶體
    if (idx < n) {
        output[idx] = result;
    }
}
```

### 重要：`__syncthreads()`

這個函數讓 Block 中的所有執行緒等待，直到都到達這個點。

⚠️ **注意**：所有執行緒必須到達 `__syncthreads()`，否則會死鎖！

```cuda
// ❌ 錯誤：只有部分執行緒執行 syncthreads
if (threadIdx.x < 16) {
    __syncthreads();  // 其他執行緒永遠不會到達！
}

// ✅ 正確：所有執行緒都會到達
__syncthreads();
if (threadIdx.x < 16) {
    // 做一些事
}
```

## 🧮 矩陣乘法優化：使用 Tiling

### 問題回顧

基本版本中，每個輸出元素需要讀取：
- A 的整行（K 個元素）
- B 的整列（K 個元素）

資料被重複讀取多次！

### 解決方案：Tiling

將矩陣分成小塊（Tile），每次只處理一個 Tile：

```
    ┌─────────────────┐
    │ Tile │ Tile │   │
B   │──────│──────│   │
    │ Tile │ Tile │   │
    └─────────────────┘

┌───────┐
│ Tile  │ ┌─────────────────┐
A │──────│ │                 │
│ Tile  │ │        C        │
└───────┘ │                 │
          └─────────────────┘
```

### Tiled 矩陣乘法

```cuda
#define TILE_SIZE 16

__global__ void matrixMulTiled(float *A, float *B, float *C,
                               int M, int K, int N) {
    // 共享記憶體：每個 Block 有自己的 tile
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // 遍歷所有 tile
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // 載入 A 的 tile
        if (row < M && t * TILE_SIZE + threadIdx.x < K) {
            tileA[threadIdx.y][threadIdx.x] =
                A[row * K + t * TILE_SIZE + threadIdx.x];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // 載入 B 的 tile
        if (t * TILE_SIZE + threadIdx.y < K && col < N) {
            tileB[threadIdx.y][threadIdx.x] =
                B[(t * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();  // 等待所有執行緒載入完成

        // 計算這個 tile 的貢獻
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads();  // 等待所有計算完成再載入下一個 tile
    }

    // 寫入結果
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

### 為什麼更快？

| 版本 | 全域記憶體讀取 | 共享記憶體讀取 |
|------|---------------|---------------|
| 基本版 | 2K 次/元素 | 0 |
| Tiled | 2K/16 次/元素 | 2K 次/元素 |

共享記憶體快 100 倍，所以整體快約 **5-10 倍**！

## 🔧 實作練習

### 範例程式

1. **shared_memory_basics.cu** - 共享記憶體基礎
2. **matrix_mul_tiled.cu** - Tiled 矩陣乘法

## 📝 今日作業

1. ✅ 理解共享記憶體的特性
2. ✅ 理解 `__syncthreads()` 的重要性
3. ✅ 執行並理解 Tiled 矩陣乘法
4. ✅ 比較與基本版的效能差異

## 🎯 練習題

### 練習 1
修改 TILE_SIZE 為 8 和 32，觀察效能變化。

### 練習 2
實作一個使用共享記憶體的向量平均計算。

---

**明天我們將學習同步與協調！** 🔄
