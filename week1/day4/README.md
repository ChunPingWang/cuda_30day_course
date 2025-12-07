# Day 4: 陣列加法 - 第一個平行運算

## 📚 今日學習目標

- 實作第一個完整的 CUDA 平行運算程式
- 學習如何在 CPU 和 GPU 之間傳輸資料
- 比較 CPU 和 GPU 的執行效能
- 理解平行運算的優勢

## 🎯 今日任務：向量加法

我們要實作：`C = A + B`

其中 A、B、C 都是包含 n 個元素的陣列（向量）。

### 為什麼這是平行運算？

傳統 CPU 做法（循序）：
```cpp
for (int i = 0; i < n; i++) {
    C[i] = A[i] + B[i];  // 一次計算一個
}
```

CUDA GPU 做法（平行）：
```cuda
// 所有元素同時計算！
C[0] = A[0] + B[0]  ⚡
C[1] = A[1] + B[1]  ⚡ 同時執行
C[2] = A[2] + B[2]  ⚡
...
```

## 📋 完整的 CUDA 程式流程

一個完整的 CUDA 程式通常包含以下步驟：

### 步驟 1：在主機端準備資料
```cpp
int *h_A, *h_B, *h_C;  // 主機端陣列
h_A = (int*)malloc(n * sizeof(int));
// 初始化資料...
```

### 步驟 2：在設備端分配記憶體
```cuda
int *d_A, *d_B, *d_C;  // 設備端陣列
cudaMalloc(&d_A, n * sizeof(int));
cudaMalloc(&d_B, n * sizeof(int));
cudaMalloc(&d_C, n * sizeof(int));
```

### 步驟 3：將資料從主機複製到設備
```cuda
cudaMemcpy(d_A, h_A, n * sizeof(int), cudaMemcpyHostToDevice);
cudaMemcpy(d_B, h_B, n * sizeof(int), cudaMemcpyHostToDevice);
```

### 步驟 4：啟動核心函數
```cuda
int threadsPerBlock = 256;
int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
vectorAdd<<<blocks, threadsPerBlock>>>(d_A, d_B, d_C, n);
```

### 步驟 5：將結果從設備複製回主機
```cuda
cudaMemcpy(h_C, d_C, n * sizeof(int), cudaMemcpyDeviceToHost);
```

### 步驟 6：釋放記憶體
```cuda
cudaFree(d_A);
cudaFree(d_B);
cudaFree(d_C);
free(h_A);
free(h_B);
free(h_C);
```

## 💻 核心函數實作

```cuda
__global__ void vectorAdd(int *a, int *b, int *c, int n) {
    // 計算全域索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 邊界檢查
    if (idx < n) {
        // 每個執行緒負責一個元素的加法
        c[idx] = a[idx] + b[idx];
    }
}
```

## 🔍 記憶體傳輸方向

`cudaMemcpy` 的第四個參數指定傳輸方向：

| 常數 | 方向 | 說明 |
|------|------|------|
| `cudaMemcpyHostToDevice` | CPU → GPU | 輸入資料 |
| `cudaMemcpyDeviceToHost` | GPU → CPU | 輸出結果 |
| `cudaMemcpyDeviceToDevice` | GPU → GPU | GPU 內部 |
| `cudaMemcpyHostToHost` | CPU → CPU | （很少用） |

## ⚡ 效能比較

查看 `vector_add_benchmark.cu`，它會比較：
- CPU 循序版本
- GPU 平行版本

對於大陣列，GPU 可以快 10-100 倍！

## 🎨 程式碼範例

### 基礎版：vector_add.cu
最簡單的向量加法實作，適合初學者。

### 進階版：vector_add_benchmark.cu
包含效能測試和 CPU 版本比較。

## 🚀 編譯與執行

```bash
# 基礎版
nvcc vector_add.cu -o vector_add
./vector_add

# 進階版（含效能測試）
nvcc vector_add_benchmark.cu -o benchmark
./benchmark
```

## 📝 今日作業

1. ✅ 執行 `vector_add.cu`
2. ✅ 執行 `vector_add_benchmark.cu`
3. ✅ 理解完整的 CUDA 程式流程
4. ✅ 完成課後練習

## 🎯 課後練習

### 練習 1：向量減法
修改程式實作 `C = A - B`

### 練習 2：純量乘法
實作 `C = k * A`，其中 k 是一個常數

提示：核心函數簽名
```cuda
__global__ void scalarMultiply(int *a, int *c, int k, int n)
```

### 練習 3：組合運算
實作 `D = A + B - C`（需要三個輸入陣列）

## 🤓 重要概念總結

### CUDA 記憶體分配函數

| 函數 | 位置 | 說明 |
|------|------|------|
| `malloc()` | CPU | C 標準函數 |
| `cudaMalloc()` | GPU | CUDA 專用 |
| `free()` | CPU | 釋放 CPU 記憶體 |
| `cudaFree()` | GPU | 釋放 GPU 記憶體 |

### 命名慣例

- `h_xxx`：主機（Host）端變數
- `d_xxx`：設備（Device）端變數

這樣可以清楚區分變數在哪裡！

## 💡 常見錯誤

### 錯誤 1：混用主機和設備指標
```cuda
// ❌ 錯誤：d_A 在 GPU 上，不能在 CPU 直接存取
d_A[0] = 10;

// ✅ 正確：使用 cudaMemcpy
int value = 10;
cudaMemcpy(d_A, &value, sizeof(int), cudaMemcpyHostToDevice);
```

### 錯誤 2：忘記檢查邊界
```cuda
// ❌ 可能存取超出範圍的記憶體
c[idx] = a[idx] + b[idx];

// ✅ 總是檢查
if (idx < n) {
    c[idx] = a[idx] + b[idx];
}
```

### 錯誤 3：忘記 cudaDeviceSynchronize
```cuda
kernel<<<blocks, threads>>>();
// ❌ CPU 不會等待，可能讀到舊資料
cudaMemcpy(h_C, d_C, ...);

// ✅ 等待 GPU 完成
kernel<<<blocks, threads>>>();
cudaDeviceSynchronize();
cudaMemcpy(h_C, d_C, ...);
```

## ❓ 思考問題

1. 為什麼需要分別在 CPU 和 GPU 分配記憶體？
2. 如果陣列有 1000 個元素，使用 `<<<4, 256>>>`，會有多少執行緒閒置？
3. 記憶體傳輸會不會成為效能瓶頸？

## 🎁 效能小貼士

- 減少 CPU-GPU 記憶體傳輸次數（傳輸很慢！）
- 盡量讓所有執行緒都有事做（避免閒置）
- 選擇適當的 Block 大小（通常 128-512）

---

**明天我們將深入學習 CUDA 記憶體管理！** 💾
