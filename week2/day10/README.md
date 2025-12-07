# Day 10: 矩陣運算入門

## 📚 今日學習目標

- 理解矩陣在 GPU 上的儲存方式
- 實作基本的矩陣運算
- 學習矩陣乘法的基本版本
- 為進階最佳化做準備

## 🧮 矩陣在記憶體中的儲存

### 行優先（Row-Major）vs 列優先（Column-Major）

CUDA/C 使用**行優先（Row-Major）**儲存：

```
矩陣 A (3×4):           記憶體排列:
┌─────────────────┐     [a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23]
│ a00 a01 a02 a03 │      ↑─────第一行─────↑ ↑─────第二行─────↑ ↑─────第三行─────↑
│ a10 a11 a12 a13 │
│ a20 a21 a22 a23 │
└─────────────────┘

索引公式: A[row][col] = A[row * cols + col]
```

## ➕ 矩陣加法

最簡單的矩陣運算，每個元素獨立計算：

```cuda
__global__ void matrixAdd(float *A, float *B, float *C,
                          int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int idx = row * cols + col;
        C[idx] = A[idx] + B[idx];
    }
}
```

### 啟動配置

```cuda
dim3 threadsPerBlock(16, 16);  // 16×16 = 256 執行緒
dim3 numBlocks((cols + 15) / 16, (rows + 15) / 16);

matrixAdd<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, rows, cols);
```

## ✖️ 矩陣乘法

矩陣乘法是 CUDA 中最經典的優化範例！

### 數學定義

```
C = A × B
C[i][j] = Σ(k=0 to K-1) A[i][k] × B[k][j]
```

### 基本版本（未優化）

```cuda
__global__ void matrixMulBasic(float *A, float *B, float *C,
                               int M, int K, int N) {
    // A: M×K, B: K×N, C: M×N
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
```

### 效能分析

對於 M=N=K=1024 的矩陣：
- 每個輸出元素需要 1024 次乘法和 1024 次加法
- 總共 1024³ ≈ 10 億次運算
- **但記憶體存取效率很低！**

### 問題在哪裡？

```
每個執行緒需要讀取:
- A 的整個第 row 行 (1024 個元素)
- B 的整個第 col 列 (1024 個元素)

如果有 1024×1024 個執行緒，每個元素會被重複讀取 1024 次！
```

## 📊 視覺化矩陣乘法

```
        B (K×N)
       ┌───────┐
       │       │
       │   *   │
       │       │
A(M×K) └───────┘
┌───┐
│   │ ┌───────┐
│ * │ │   C   │  C[i][j] = A 的第 i 行 · B 的第 j 列
│   │ │ (M×N) │
└───┘ └───────┘
```

## 🔧 實作練習

### 範例程式

1. **matrix_add.cu** - 矩陣加法
2. **matrix_mul_basic.cu** - 矩陣乘法基本版

### 編譯與執行

```bash
nvcc matrix_add.cu -o matrix_add
./matrix_add

nvcc matrix_mul_basic.cu -o matrix_mul
./matrix_mul
```

## 📝 今日作業

1. ✅ 理解矩陣在記憶體中的儲存方式
2. ✅ 實作矩陣加法
3. ✅ 理解矩陣乘法的基本版本
4. ✅ 完成練習題

## 🎯 練習題

### 練習 1
實作矩陣減法：`C = A - B`

### 練習 2
實作純量矩陣乘法：`C = k × A`

### 練習 3
計算矩陣乘法的記憶體存取次數（對於 N×N 矩陣）

## 💡 預告

明天我們將學習**共享記憶體**，這是優化矩陣乘法的關鍵技術！

使用共享記憶體後，矩陣乘法效能可以提升 **5-10 倍**！

---

**明天我們將學習共享記憶體（Shared Memory）！** 💾
