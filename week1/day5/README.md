# Day 5: CUDA 記憶體管理基礎

## 📚 今日學習目標

- 理解 CUDA 的記憶體層次結構
- 學習各種記憶體類型的特點和用途
- 掌握記憶體分配和傳輸的最佳實踐
- 理解統一記憶體（Unified Memory）的概念

## 🧠 CUDA 記憶體層次結構

GPU 有多種記憶體類型，各有不同的特性：

```
┌─────────────────────────────────────────┐
│              GPU 晶片                    │
│  ┌─────────────────────────────────┐    │
│  │         共享記憶體               │    │ ← 每個 Block 共享
│  │    (Shared Memory)              │    │   速度很快
│  ├─────────────────────────────────┤    │
│  │  暫存器 │ 暫存器 │ 暫存器 │ ... │    │ ← 每個 Thread 私有
│  │(Register)                       │    │   最快的記憶體
│  └─────────────────────────────────┘    │
├─────────────────────────────────────────┤
│  L1/L2 快取 (Cache)                     │ ← 自動管理
├─────────────────────────────────────────┤
│  常數記憶體 (Constant Memory)           │ ← 唯讀，有快取
├─────────────────────────────────────────┤
│  紋理記憶體 (Texture Memory)            │ ← 特殊用途
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│  全域記憶體 (Global Memory)              │ ← 最大，但最慢
│  (這就是你的顯示卡 VRAM - 8GB)           │
└─────────────────────────────────────────┘
```

## 📊 記憶體類型比較

| 記憶體類型 | 速度 | 大小 | 作用域 | 生命週期 |
|-----------|------|------|--------|----------|
| 暫存器 | ⚡⚡⚡⚡⚡ | 非常小 | 單一 Thread | Kernel |
| 共享記憶體 | ⚡⚡⚡⚡ | ~48KB/Block | 單一 Block | Kernel |
| 常數記憶體 | ⚡⚡⚡ | 64KB | 所有 Thread | 應用程式 |
| 全域記憶體 | ⚡ | 數 GB | 所有 Thread | 應用程式 |

## 💾 全域記憶體管理

### 分配記憶體：cudaMalloc

```cuda
// 語法
cudaError_t cudaMalloc(void **devPtr, size_t size);

// 範例
float *d_array;
cudaMalloc(&d_array, n * sizeof(float));
```

### 釋放記憶體：cudaFree

```cuda
cudaFree(d_array);
```

### 資料傳輸：cudaMemcpy

```cuda
// 語法
cudaError_t cudaMemcpy(void *dst, const void *src, size_t count,
                       cudaMemcpyKind kind);

// 範例：CPU → GPU
cudaMemcpy(d_array, h_array, n * sizeof(float), cudaMemcpyHostToDevice);

// 範例：GPU → CPU
cudaMemcpy(h_result, d_result, n * sizeof(float), cudaMemcpyDeviceToHost);
```

### 初始化記憶體：cudaMemset

```cuda
// 將記憶體設為 0
cudaMemset(d_array, 0, n * sizeof(float));
```

## 🌟 統一記憶體（Unified Memory）

CUDA 6.0 之後引入了統一記憶體，讓記憶體管理更簡單！

### 傳統方式 vs 統一記憶體

**傳統方式：**
```cuda
float *h_data, *d_data;
h_data = (float*)malloc(n * sizeof(float));
cudaMalloc(&d_data, n * sizeof(float));
cudaMemcpy(d_data, h_data, n * sizeof(float), cudaMemcpyHostToDevice);
// ... 執行核心函數 ...
cudaMemcpy(h_data, d_data, n * sizeof(float), cudaMemcpyDeviceToHost);
cudaFree(d_data);
free(h_data);
```

**統一記憶體：**
```cuda
float *data;
cudaMallocManaged(&data, n * sizeof(float));
// CPU 和 GPU 都可以直接使用 data！
// ... 執行核心函數 ...
cudaDeviceSynchronize();  // 確保 GPU 完成
// 直接在 CPU 讀取結果
cudaFree(data);
```

### cudaMallocManaged

```cuda
cudaError_t cudaMallocManaged(void **devPtr, size_t size);
```

- 分配的記憶體 CPU 和 GPU 都可以存取
- 系統自動處理資料傳輸
- 簡化程式碼，減少錯誤
- 適合原型開發和學習

## ⚠️ 錯誤處理

CUDA 函數會回傳錯誤碼，我們應該檢查：

```cuda
cudaError_t err = cudaMalloc(&d_array, n * sizeof(float));
if (err != cudaSuccess) {
    printf("CUDA 錯誤: %s\n", cudaGetErrorString(err));
    exit(1);
}
```

### 實用的錯誤檢查巨集

```cuda
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s at line %d\n", \
               cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
}

// 使用方式
CHECK_CUDA(cudaMalloc(&d_array, n * sizeof(float)));
```

## 🔧 實作練習

查看並執行範例程式：

### 1. memory_basics.cu
展示基本的記憶體管理操作。

### 2. unified_memory.cu
使用統一記憶體簡化程式碼。

### 3. memory_transfer.cu
測量記憶體傳輸的效能。

## 📝 今日作業

1. ✅ 執行並理解 `memory_basics.cu`
2. ✅ 執行並理解 `unified_memory.cu`
3. ✅ 比較兩種記憶體管理方式的差異
4. ✅ 完成練習題

## 🎯 練習題

### 練習 1
修改 `memory_basics.cu`，加入完整的錯誤檢查。

### 練習 2
使用統一記憶體實作向量乘法 `C = A * B`（元素對元素相乘）。

## 💡 記憶體使用建議

### 何時使用全域記憶體
- 資料量大
- 需要在多個 Kernel 之間共享資料
- 資料需要傳回 CPU

### 何時使用共享記憶體（下週會學）
- 同一個 Block 內的 Thread 需要協作
- 資料會被重複使用
- 需要高速存取

### 何時使用統一記憶體
- 原型開發和學習
- 不確定最佳的記憶體配置
- 程式碼可讀性優先

## ❓ 思考問題

1. 為什麼 GPU 的全域記憶體比暫存器和共享記憶體慢？
2. 統一記憶體的優點和缺點是什麼？
3. 為什麼我們需要 `cudaDeviceSynchronize()` 在使用統一記憶體時？

## 🎁 進階知識

### 記憶體對齊（Memory Alignment）
- CUDA 喜歡對齊到 32, 64, 或 128 bytes
- 對齊的存取效率更高

### Pinned Memory（固定記憶體）
```cuda
float *h_pinned;
cudaMallocHost(&h_pinned, n * sizeof(float));  // 分配 pinned memory
// 傳輸速度比普通記憶體快 2-3 倍！
cudaFreeHost(h_pinned);
```

---

**明天我們將學習執行緒索引與資料對應的進階技巧！** 🎯
