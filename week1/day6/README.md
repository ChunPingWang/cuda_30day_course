# Day 6: 執行緒索引與資料對應進階

## 📚 今日學習目標

- 熟練一維、二維索引計算
- 學習處理任意大小的資料
- 理解 Stride Pattern（步幅模式）
- 實作 2D 矩陣操作

## 📐 複習：一維索引

```cuda
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

### 圖示

```
Block 0              Block 1              Block 2
┌────────────────┐  ┌────────────────┐  ┌────────────────┐
│ T0  T1  T2  T3 │  │ T0  T1  T2  T3 │  │ T0  T1  T2  T3 │
│ (0) (1) (2) (3)│  │ (4) (5) (6) (7)│  │ (8) (9)(10)(11)│
└────────────────┘  └────────────────┘  └────────────────┘
     idx = blockIdx.x * 4 + threadIdx.x
```

## 🎯 處理大量資料：Grid-Stride Loop

當資料量大於執行緒數量時，每個執行緒可以處理多個元素！

### 問題

假設有 10,000 個元素，但我們只想用 256 個執行緒：
- 傳統方式：啟動 40 個 Block
- Grid-Stride：每個執行緒處理約 40 個元素

### Grid-Stride Loop 模式

```cuda
__global__ void processArray(float *arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;  // 總執行緒數

    // 每個執行緒用步幅方式處理多個元素
    for (int i = idx; i < n; i += stride) {
        arr[i] = arr[i] * 2.0f;
    }
}
```

### 視覺化

```
元素:     [0] [1] [2] [3] [4] [5] [6] [7] [8] [9] [10] [11] ...
           ↓   ↓   ↓   ↓
Thread:    0   1   2   3   0   1   2   3   0   1    2    3  ...
                          ↑               ↑
                       stride          stride
```

### 優點

1. 固定的執行緒配置
2. 更好的記憶體存取模式
3. 適用於任意大小的資料

## 🎲 二維索引

### 為什麼需要二維？

處理圖片、矩陣等二維資料時，二維索引更直觀！

### 二維 Grid 和 Block

```cuda
// 定義二維 Block（每個 Block 16×16 = 256 個執行緒）
dim3 threadsPerBlock(16, 16);

// 定義二維 Grid
dim3 numBlocks((width + 15) / 16, (height + 15) / 16);

// 啟動核心函數
myKernel<<<numBlocks, threadsPerBlock>>>();
```

### 二維索引計算

```cuda
__global__ void process2D(float *data, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        // 將 2D 座標轉換為 1D 索引（行優先）
        int idx = y * width + x;
        data[idx] = data[idx] * 2.0f;
    }
}
```

### 圖示：2D 資料對應

```
           x →
     ┌───────────────────────────┐
   y │ (0,0) (1,0) (2,0) (3,0) ...
   ↓ │ (0,1) (1,1) (2,1) (3,1) ...
     │ (0,2) (1,2) (2,2) (3,2) ...
     │  ...   ...   ...   ...
     └───────────────────────────┘

1D 索引 = y * width + x
例如: (2,1) → 1 * width + 2
```

## 🖼️ 實例：圖片處理

```cuda
__global__ void brighten(unsigned char *image,
                         int width, int height,
                         int brightness) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;

        // 增加亮度（限制在 0-255）
        int newValue = image[idx] + brightness;
        image[idx] = (newValue > 255) ? 255 :
                     (newValue < 0) ? 0 : newValue;
    }
}
```

## 🧮 矩陣操作

### 矩陣轉置

```cuda
__global__ void transpose(float *input, float *output,
                          int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        // 輸入: (x, y) 位置
        int inputIdx = y * width + x;

        // 輸出: (y, x) 位置（轉置後寬高交換）
        int outputIdx = x * height + y;

        output[outputIdx] = input[inputIdx];
    }
}
```

## 🔧 實作練習

### 範例程式

1. **stride_pattern.cu** - Grid-Stride Loop 示範
2. **2d_indexing.cu** - 二維索引操作
3. **matrix_ops.cu** - 矩陣操作實例

### 編譯與執行

```bash
nvcc stride_pattern.cu -o stride_pattern
./stride_pattern

nvcc 2d_indexing.cu -o 2d_indexing
./2d_indexing
```

## 📝 今日作業

1. ✅ 理解 Grid-Stride Loop 模式
2. ✅ 練習二維索引計算
3. ✅ 執行所有範例程式
4. ✅ 完成練習題

## 🎯 練習題

### 練習 1：矩陣加法
實作兩個矩陣相加：`C[i][j] = A[i][j] + B[i][j]`

### 練習 2：矩陣縮放
將矩陣中每個元素乘以一個常數。

### 練習 3：邊緣填充
將矩陣的邊緣（第一行、最後一行、第一列、最後一列）設為 0。

提示：使用 `if` 條件檢查座標

## 🤓 重要公式總結

### 一維索引
```cuda
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

### 二維索引
```cuda
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int idx = y * width + x;
```

### Grid-Stride Loop
```cuda
int idx = blockIdx.x * blockDim.x + threadIdx.x;
int stride = blockDim.x * gridDim.x;
for (int i = idx; i < n; i += stride) {
    // 處理第 i 個元素
}
```

### Block 數量計算
```cuda
// 一維
int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

// 二維
dim3 blocks((width + blockDim.x - 1) / blockDim.x,
            (height + blockDim.y - 1) / blockDim.y);
```

## ❓ 思考問題

1. Grid-Stride Loop 比傳統方式有什麼優勢？
2. 為什麼 2D 索引轉 1D 要用 `y * width + x`？
3. 如果資料是列優先（column-major）而非行優先，索引公式要怎麼改？

## 💡 小技巧

- Block 大小通常選擇 16×16 或 32×32（圖像處理）
- 8×8 適合小型操作
- 總是要做邊界檢查！

---

**明天是週末練習與複習！我們會整合本週學到的所有知識。** 📚
