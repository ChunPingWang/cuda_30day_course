# Day 3: 理解核心函數與執行緒索引

## 📚 今日學習目標

- 深入理解執行緒（Thread）、區塊（Block）和網格（Grid）
- 學會使用內建變數來識別執行緒
- 理解執行緒索引的計算方法
- 讓每個執行緒處理不同的資料

## 🧵 執行緒階層結構

CUDA 有三層組織結構：

```
Grid（網格）
 └─ Block（區塊）
     └─ Thread（執行緒）
```

### 用班級來比喻

想像你是校長，要管理整個學校：

- **Grid（網格）** = 整個學校
- **Block（區塊）** = 每個班級
- **Thread（執行緒）** = 每個學生

你可以說：「3 年 2 班的 15 號同學」來精確指定一個學生！

在 CUDA 中：
```cuda
myKernel<<<3, 32>>>();
```
表示：3 個 Block，每個 Block 有 32 個 Thread

## 🔢 內建變數：識別執行緒的身分證

CUDA 提供了幾個特殊的內建變數，讓每個執行緒知道「我是誰」：

### 重要的內建變數

| 變數 | 說明 | 類型 |
|------|------|------|
| `threadIdx.x` | 執行緒在 Block 中的索引 | `uint3` |
| `blockIdx.x` | Block 在 Grid 中的索引 | `uint3` |
| `blockDim.x` | 每個 Block 的執行緒數量 | `dim3` |
| `gridDim.x` | Grid 中的 Block 數量 | `dim3` |

### 一維索引計算

要計算執行緒的全域索引（在所有執行緒中的位置）：

```cuda
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

### 圖解說明

假設我們有 `<<<3, 4>>>`（3 個 Block，每個 4 個 Thread）：

```
Block 0:        Block 1:        Block 2:
Thread 0 (0)    Thread 0 (4)    Thread 0 (8)
Thread 1 (1)    Thread 1 (5)    Thread 1 (9)
Thread 2 (2)    Thread 2 (6)    Thread 2 (10)
Thread 3 (3)    Thread 3 (7)    Thread 3 (11)
```

括號內是全域索引！

## 💡 實例：讓每個執行緒報上自己的編號

查看 `thread_index.cu`，這個程式展示如何使用執行緒索引。

### 關鍵程式碼

```cuda
__global__ void printThreadInfo() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Block %d, Thread %d => Global Index: %d\n",
           blockIdx.x, threadIdx.x, idx);
}
```

## 🎯 實際應用：處理陣列

每個執行緒處理陣列中的一個元素！

```cuda
__global__ void doubleArray(int *arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 確保不超出陣列範圍
    if (idx < n) {
        arr[idx] = arr[idx] * 2;
    }
}
```

### 為什麼需要檢查 `idx < n`？

因為執行緒數量可能比陣列元素多！

例如：
- 陣列有 100 個元素
- 我們啟動 `<<<4, 32>>>`（128 個執行緒）
- 索引 100-127 的執行緒要跳過，避免存取超出範圍的記憶體

## 📊 維度：不只是一維！

CUDA 支援最多三維的組織：

```cuda
dim3 blocks(2, 2, 1);   // 2x2x1 的 Block 網格
dim3 threads(4, 4, 1);  // 4x4x1 的 Thread 區塊
myKernel<<<blocks, threads>>>();
```

這時可以用 `.x`、`.y`、`.z` 來存取：
```cuda
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
```

這對處理圖片（2D）或 3D 模擬特別有用！

## 🔧 編譯與執行

### CUDA 編譯

#### thread_index.cu - 執行緒索引示範

**Windows（cmd）：**

```cmd
nvcc -allow-unsupported-compiler -Wno-deprecated-gpu-targets -Xcompiler "/wd4819" -o thread_index.exe thread_index.cu
thread_index.exe
```

**Windows（PowerShell）：**

```powershell
nvcc -allow-unsupported-compiler -Wno-deprecated-gpu-targets -Xcompiler "/wd4819" -o thread_index.exe thread_index.cu
.\thread_index.exe
```

**WSL / Linux：**

```bash
nvcc -Wno-deprecated-gpu-targets -o thread_index thread_index.cu
./thread_index
```

#### array_index.cu - 陣列索引示範

**Windows（cmd）：**

```cmd
nvcc -allow-unsupported-compiler -Wno-deprecated-gpu-targets -Xcompiler "/wd4819" -o array_index.exe array_index.cu
array_index.exe
```

**Windows（PowerShell）：**

```powershell
nvcc -allow-unsupported-compiler -Wno-deprecated-gpu-targets -Xcompiler "/wd4819" -o array_index.exe array_index.cu
.\array_index.exe
```

**WSL / Linux：**

```bash
nvcc -Wno-deprecated-gpu-targets -o array_index array_index.cu
./array_index
```

### Python 等效

```python
import cupy as cp
# CuPy 使用 RawKernel 展示執行緒索引
kernel = cp.RawKernel(r'''
extern "C" __global__ void showIndex() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Block %d, Thread %d => Global Index: %d\n",
           blockIdx.x, threadIdx.x, idx);
}
''', 'showIndex')
kernel((3,), (4,), ())
cp.cuda.Stream.null.synchronize()
```

## 🚀 實作練習

觀察 `thread_index.cu` 不同配置的輸出，以及 `array_index.cu` 如何用執行緒索引來處理陣列。

## 📝 今日作業

1. ✅ 理解執行緒索引的計算公式
2. ✅ 執行並理解 `thread_index.cu`
3. ✅ 執行並理解 `array_index.cu`
4. ✅ 完成課後練習

## 🎯 課後練習

修改 `array_index.cu`，讓它計算：
- 每個執行緒將自己的全域索引存入陣列
- 例如：`array[0] = 0, array[1] = 1, array[2] = 2, ...`

## 🤓 重要公式

### 一維全域索引
```cuda
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

### 二維全域索引
```cuda
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
```

### 計算需要的 Block 數量
```cuda
int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
```
這個公式確保有足夠的執行緒處理 n 個元素。

## ❓ 思考問題

1. 如果有 1000 個元素，每個 Block 有 256 個執行緒，需要幾個 Block？
2. 為什麼最後的執行緒需要檢查 `idx < n`？
3. 二維索引在什麼情況下比一維索引更方便？

## 💡 小技巧

- 通常選擇 32 的倍數作為 Block 大小（如 128, 256, 512）
- 原因是 GPU 的 Warp 大小是 32（我們第二週會學到）
- 最常用的配置：`<<<(n+255)/256, 256>>>`

---

**明天我們將實作第一個真正的平行運算：陣列加法！** ➕
