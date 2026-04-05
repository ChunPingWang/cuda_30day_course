# Day 7: 週末練習與複習

## 📚 本週學習回顧

恭喜你完成了 CUDA 學習的第一週！讓我們來回顧一下這週學到的內容。

### 學習進度

| 天數 | 主題 | 核心概念 |
|------|------|----------|
| Day 1 | CUDA 簡介 | GPU vs CPU、環境設定 |
| Day 2 | Hello World | Kernel、`__global__`、`<<<>>>` |
| Day 3 | 執行緒索引 | threadIdx、blockIdx、blockDim |
| Day 4 | 向量加法 | 完整 CUDA 程式流程 |
| Day 5 | 記憶體管理 | cudaMalloc、cudaMemcpy、統一記憶體 |
| Day 6 | 進階索引 | Grid-Stride Loop、2D 索引 |

## 🧠 核心概念總結

### 1. CUDA 程式架構

```
主機程式 (CPU)                   設備程式 (GPU)
    │                               │
    ├── 分配記憶體                  │
    ├── 準備資料                    │
    ├── 複製資料到 GPU ──────────→ │
    │                               │
    ├── 啟動 Kernel ──────────────→├── 執行平行運算
    │                               │
    ├── 等待完成                    │
    ├── 複製結果回 CPU ←────────── │
    ├── 處理結果                    │
    └── 釋放記憶體                  │
```

### 2. 執行緒組織

```
Grid
 ├── Block 0
 │    ├── Thread 0
 │    ├── Thread 1
 │    └── ...
 ├── Block 1
 │    ├── Thread 0
 │    ├── Thread 1
 │    └── ...
 └── ...
```

### 3. 重要函數

| 函數 | 用途 |
|------|------|
| `cudaMalloc()` | 在 GPU 分配記憶體 |
| `cudaFree()` | 釋放 GPU 記憶體 |
| `cudaMemcpy()` | CPU ↔ GPU 資料傳輸 |
| `cudaMallocManaged()` | 分配統一記憶體 |
| `cudaDeviceSynchronize()` | 等待 GPU 完成 |

### 4. 索引計算

```cuda
// 一維
int idx = blockIdx.x * blockDim.x + threadIdx.x;

// 二維
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int idx = y * width + x;
```

## 🎯 綜合練習

### 練習 1：向量運算（難度：⭐）

實作一個程式，計算向量的每個元素的平方：`B[i] = A[i] * A[i]`

### 練習 2：向量點積（難度：⭐⭐）

計算兩個向量的點積：`result = A[0]*B[0] + A[1]*B[1] + ...`

提示：這需要將每個執行緒的結果加總。可以先讓每個執行緒計算一部分，然後在 CPU 上加總。

### 練習 3：矩陣操作（難度：⭐⭐）

實作矩陣縮放：將矩陣中每個元素乘以一個常數 k。

### 練習 4：陣列統計（難度：⭐⭐⭐）

找出陣列中的最大值。

提示：這需要比較不同執行緒的結果，稍後會學到更好的方法（平行歸約）。

## 📝 練習解答

查看 `exercises/` 目錄中的解答：
- `ex1_square.cu` - 向量平方
- `ex2_dot_product.cu` - 向量點積
- `ex3_matrix_scale.cu` - 矩陣縮放
- `ex4_find_max.cu` - 陣列最大值

## 🔧 編譯與執行

### CUDA 編譯

**Windows（cmd）：**

```cmd
nvcc -allow-unsupported-compiler -Wno-deprecated-gpu-targets -Xcompiler "/wd4819" -o ex1_square.exe ex1_square.cu
ex1_square.exe

nvcc -allow-unsupported-compiler -Wno-deprecated-gpu-targets -Xcompiler "/wd4819" -o ex2_dot_product.exe ex2_dot_product.cu
ex2_dot_product.exe

nvcc -allow-unsupported-compiler -Wno-deprecated-gpu-targets -Xcompiler "/wd4819" -o ex3_matrix_scale.exe ex3_matrix_scale.cu
ex3_matrix_scale.exe

nvcc -allow-unsupported-compiler -Wno-deprecated-gpu-targets -Xcompiler "/wd4819" -o ex4_find_max.exe ex4_find_max.cu
ex4_find_max.exe
```

**Windows（PowerShell）：**

```powershell
nvcc -allow-unsupported-compiler -Wno-deprecated-gpu-targets -Xcompiler "/wd4819" -o ex1_square.exe ex1_square.cu
.\ex1_square.exe

nvcc -allow-unsupported-compiler -Wno-deprecated-gpu-targets -Xcompiler "/wd4819" -o ex2_dot_product.exe ex2_dot_product.cu
.\ex2_dot_product.exe

nvcc -allow-unsupported-compiler -Wno-deprecated-gpu-targets -Xcompiler "/wd4819" -o ex3_matrix_scale.exe ex3_matrix_scale.cu
.\ex3_matrix_scale.exe

nvcc -allow-unsupported-compiler -Wno-deprecated-gpu-targets -Xcompiler "/wd4819" -o ex4_find_max.exe ex4_find_max.cu
.\ex4_find_max.exe
```

**WSL / Linux：**

```bash
nvcc -Wno-deprecated-gpu-targets -o ex1_square ex1_square.cu
./ex1_square

nvcc -Wno-deprecated-gpu-targets -o ex2_dot_product ex2_dot_product.cu
./ex2_dot_product

nvcc -Wno-deprecated-gpu-targets -o ex3_matrix_scale ex3_matrix_scale.cu
./ex3_matrix_scale

nvcc -Wno-deprecated-gpu-targets -o ex4_find_max ex4_find_max.cu
./ex4_find_max
```

### Python 等效

```python
import cupy as cp

# 練習 1：陣列平方
a = cp.array([1, 2, 3, 4, 5], dtype=cp.float32)
squared = a ** 2

# 練習 2：內積
a = cp.array([1, 2, 3], dtype=cp.float32)
b = cp.array([4, 5, 6], dtype=cp.float32)
dot = cp.dot(a, b)

# 練習 3：矩陣縮放
matrix = cp.ones((4, 4), dtype=cp.float32)
scaled = matrix * 2.5

# 練習 4：找最大值
data = cp.random.rand(1000, dtype=cp.float32)
max_val = cp.max(data)
```

## 🔍 自我測驗

回答以下問題來測試你的理解：

### 問題 1
```cuda
kernel<<<4, 8>>>();
```
這個配置會啟動多少個執行緒？

<details>
<summary>查看答案</summary>
4 × 8 = 32 個執行緒
</details>

### 問題 2
如果有 1000 個元素，使用 `<<<4, 256>>>` 配置，會有多少執行緒閒置？

<details>
<summary>查看答案</summary>
總執行緒 = 4 × 256 = 1024
閒置 = 1024 - 1000 = 24 個執行緒
</details>

### 問題 3
這段程式碼有什麼問題？
```cuda
int *d_array;
cudaMalloc(&d_array, 100 * sizeof(int));
d_array[0] = 10;  // ← 這行
```

<details>
<summary>查看答案</summary>
不能在 CPU 端直接存取 GPU 記憶體！
需要使用 cudaMemcpy 或使用統一記憶體。
</details>

### 問題 4
`cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice)` 中：
- src 在哪裡？
- dst 在哪裡？

<details>
<summary>查看答案</summary>
- src 在 CPU（Host）
- dst 在 GPU（Device）
</details>

## 💡 學習建議

### 做得好的地方
- ✅ 完成了所有範例程式
- ✅ 理解了基本的記憶體管理
- ✅ 學會了執行緒索引計算

### 需要加強的地方
如果某些概念還不清楚，建議：
1. 重新閱讀該天的教材
2. 多執行幾次範例程式
3. 嘗試修改程式觀察結果
4. 畫圖幫助理解執行緒組織

## 🚀 下週預告

第二週我們將學習：
- Grid、Block、Thread 的進階概念
- 共享記憶體（Shared Memory）
- 同步（Synchronization）
- 矩陣乘法（重要的優化範例）

## 📋 本週專案

完成一個綜合專案：**圖片亮度調整**

查看 `week1_project/` 目錄，這個專案整合了：
- 記憶體管理
- 2D 索引
- 核心函數設計

## ✅ 完成清單

在繼續下週學習前，確保你能：

- [ ] 解釋 CPU 和 GPU 的區別
- [ ] 寫出一個簡單的核心函數
- [ ] 使用 cudaMalloc 和 cudaFree
- [ ] 使用 cudaMemcpy 傳輸資料
- [ ] 計算一維執行緒索引
- [ ] 計算二維執行緒索引
- [ ] 使用 Grid-Stride Loop

---

**恭喜完成第一週！休息一下，下週我們將學習更多進階技巧！** 🎉
