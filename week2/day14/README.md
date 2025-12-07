# Day 14: 週末專題 - 完整的矩陣乘法優化

## 📚 今日學習目標

- 整合本週學到的所有技術
- 實作多個版本的矩陣乘法
- 進行完整的效能比較
- 理解優化的實際效果

## 🎯 專題目標

實作並比較五個版本的矩陣乘法：

1. **CPU 版本**（基準）
2. **GPU 基本版本**
3. **GPU + 共享記憶體**
4. **GPU + 共享記憶體 + 迴圈展開**
5. **cuBLAS**（官方函式庫）

## 📊 效能比較框架

我們將測量並比較：
- 執行時間（ms）
- GFLOPS（每秒十億次浮點運算）
- 相對於 CPU 的加速比

## 💻 完整程式碼

查看 `matrix_mul_complete.cu`，包含所有版本的實作和效能比較。

### 版本 1：CPU 基準

```cpp
void matrixMulCPU(float *A, float *B, float *C, int M, int K, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}
```

### 版本 2：GPU 基本版

```cuda
__global__ void matrixMulBasic(float *A, float *B, float *C,
                               int M, int K, int N) {
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

### 版本 3：使用共享記憶體

```cuda
#define TILE_SIZE 16

__global__ void matrixMulShared(float *A, float *B, float *C,
                                int M, int K, int N) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // 載入 tiles
        if (row < M && t * TILE_SIZE + threadIdx.x < K)
            tileA[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        if (t * TILE_SIZE + threadIdx.y < K && col < N)
            tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++)
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}
```

### 版本 4：加上迴圈展開

```cuda
__global__ void matrixMulUnrolled(float *A, float *B, float *C,
                                  int M, int K, int N) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // 載入（與版本 3 相同）
        // ...

        __syncthreads();

        // 展開內層迴圈
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++)
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}
```

### 版本 5：使用 cuBLAS

```cpp
#include <cublas_v2.h>

void matrixMulCuBLAS(float *A, float *B, float *C, int M, int K, int N) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f, beta = 0.0f;

    // cuBLAS 使用列優先，所以參數順序調整
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K, &alpha, B, N, A, K, &beta, C, N);

    cublasDestroy(handle);
}
```

## 📈 預期效能結果

對於 1024×1024 矩陣：

| 版本 | 時間 | GFLOPS | 加速比 |
|------|------|--------|--------|
| CPU | ~3000 ms | ~0.7 | 1x |
| GPU 基本 | ~50 ms | ~40 | ~60x |
| GPU 共享記憶體 | ~15 ms | ~140 | ~200x |
| GPU 展開 | ~12 ms | ~180 | ~250x |
| cuBLAS | ~3 ms | ~700 | ~1000x |

## 🔧 編譯與執行

```bash
# 編譯（需要連結 cuBLAS）
nvcc matrix_mul_complete.cu -o matrix_mul_complete -lcublas

# 執行
./matrix_mul_complete
```

## 📝 本週總結

### 學到的技術

1. **Warp 和執行模型**
2. **記憶體合併**
3. **共享記憶體**
4. **同步機制**
5. **原子操作**
6. **錯誤處理**

### 優化技巧

1. 減少全域記憶體存取
2. 使用共享記憶體快取資料
3. 確保記憶體合併
4. 避免 Warp 分歧
5. 選擇適當的 Block 大小

## 🎯 挑戰練習

1. 嘗試不同的 TILE_SIZE（8, 16, 32）
2. 實作 2×2 的輸出 tile（每個執行緒計算 4 個元素）
3. 測試不同大小的矩陣

## 💡 下週預告

第三週我們將學習：
- 記憶體優化進階技巧
- 平行歸約（Reduction）
- 掃描（Scan）
- 效能分析工具

---

**恭喜完成第二週！🎉 休息一下，下週見！**
