# Day 28: 期末專題（一）- 專題規劃與架構設計

## 今日學習目標

- 規劃期末專題
- 設計程式架構
- 準備開發環境
- 建立專案框架

## 期末專題選項

選擇以下其中一個專題完成：

### 專題 A：GPU 加速的圖像處理器

實作一個完整的圖像處理應用：
- 多種濾波器（模糊、銳化、邊緣檢測）
- 直方圖均衡化
- 效能比較報告

### 專題 B：矩陣運算庫

實作高效的矩陣運算：
- 矩陣乘法（多種優化版本）
- 矩陣轉置
- 矩陣求逆
- 與 cuBLAS 效能比較

### 專題 C：簡易神經網路

實作基本的神經網路運算：
- 全連接層前向傳播
- 激活函數（ReLU、Sigmoid）
- Softmax 輸出
- 簡單的 MNIST 推論

## 專題 A：圖像處理器 - 詳細規劃

### 功能需求

```
輸入圖像 → [GPU 處理] → 輸出圖像
              │
              ├─ 高斯模糊
              ├─ 銳化
              ├─ 邊緣檢測（Sobel）
              ├─ 灰階轉換
              └─ 直方圖均衡化
```

### 程式架構

```
image_processor/
├── include/
│   ├── image.h           # 圖像結構定義
│   ├── filters.h         # 濾波器宣告
│   └── utils.h           # 工具函數
├── src/
│   ├── main.cu           # 主程式
│   ├── filters.cu        # GPU 濾波器實作
│   ├── image_io.cu       # 圖像讀寫
│   └── benchmark.cu      # 效能測試
├── data/
│   └── test_image.ppm    # 測試圖像
└── Makefile
```

### 資料結構

```cuda
// include/image.h
typedef struct {
    unsigned char *data;    // 像素資料
    int width;              // 寬度
    int height;             // 高度
    int channels;           // 通道數 (1=灰階, 3=RGB)
} Image;

// GPU 記憶體版本
typedef struct {
    unsigned char *d_data;  // GPU 上的像素資料
    int width;
    int height;
    int channels;
} GpuImage;
```

### 核心函數宣告

```cuda
// include/filters.h

// 高斯模糊
void gaussianBlur(GpuImage *src, GpuImage *dst, int kernelSize, float sigma);

// Sobel 邊緣檢測
void sobelEdgeDetection(GpuImage *src, GpuImage *dst);

// 銳化
void sharpen(GpuImage *src, GpuImage *dst, float strength);

// 灰階轉換
void rgbToGray(GpuImage *src, GpuImage *dst);

// 直方圖均衡化
void histogramEqualization(GpuImage *src, GpuImage *dst);
```

## 專題 B：矩陣運算庫 - 詳細規劃

### 功能需求

```
矩陣運算庫
├── 基本運算
│   ├── 矩陣加法
│   ├── 矩陣減法
│   └── 純量乘法
├── 進階運算
│   ├── 矩陣乘法（多版本）
│   │   ├── 基本版
│   │   ├── 共享記憶體版
│   │   └── cuBLAS 版
│   ├── 矩陣轉置
│   └── 矩陣求逆（LU 分解）
└── 效能測試
```

### 程式架構

```
matrix_lib/
├── include/
│   ├── matrix.h          # 矩陣結構
│   ├── operations.h      # 運算宣告
│   └── cublas_wrapper.h  # cuBLAS 包裝
├── src/
│   ├── main.cu           # 主程式/測試
│   ├── matrix_basic.cu   # 基本運算
│   ├── matrix_mul.cu     # 矩陣乘法
│   ├── matrix_transpose.cu
│   └── benchmark.cu
└── Makefile
```

### 資料結構

```cuda
// include/matrix.h
typedef struct {
    float *data;     // 資料（行優先）
    int rows;        // 列數
    int cols;        // 欄數
} Matrix;

typedef struct {
    float *d_data;   // GPU 資料
    int rows;
    int cols;
} GpuMatrix;

// 建構/解構函數
Matrix* createMatrix(int rows, int cols);
void freeMatrix(Matrix *m);
GpuMatrix* createGpuMatrix(int rows, int cols);
void freeGpuMatrix(GpuMatrix *m);
void copyToGpu(Matrix *src, GpuMatrix *dst);
void copyFromGpu(GpuMatrix *src, Matrix *dst);
```

## 專題 C：簡易神經網路 - 詳細規劃

### 功能需求

```
神經網路推論
├── 層類型
│   ├── 全連接層
│   └── 激活層
├── 激活函數
│   ├── ReLU
│   ├── Sigmoid
│   └── Softmax
└── 應用
    └── MNIST 手寫數字辨識
```

### 程式架構

```
neural_network/
├── include/
│   ├── layer.h           # 層定義
│   ├── activations.h     # 激活函數
│   └── network.h         # 網路結構
├── src/
│   ├── main.cu           # 主程式
│   ├── dense_layer.cu    # 全連接層
│   ├── activations.cu    # 激活函數
│   └── mnist_loader.cu   # MNIST 資料載入
├── data/
│   ├── weights.bin       # 預訓練權重
│   └── mnist/            # MNIST 資料集
└── Makefile
```

### 資料結構

```cuda
// include/layer.h
typedef struct {
    float *d_weights;    // 權重 [out_features x in_features]
    float *d_bias;       // 偏置 [out_features]
    int in_features;
    int out_features;
} DenseLayer;

// include/network.h
typedef struct {
    DenseLayer *layers;
    int num_layers;
} Network;

// 核心運算
void denseForward(DenseLayer *layer, float *d_input, float *d_output, int batch_size);
void relu(float *d_data, int n);
void sigmoid(float *d_data, int n);
void softmax(float *d_data, int batch_size, int num_classes);
```

## 開發時程

| 天數 | 工作內容 |
|------|----------|
| Day 28 | 專題規劃與架構設計 |
| Day 29 | 核心功能實作 |
| Day 30 | 優化、測試與文件 |

## 今日作業

1. 選擇一個專題
2. 建立專案目錄結構
3. 編寫標頭檔和資料結構
4. 實作基本的記憶體管理函數

## 範例：建立專案框架

```bash
# 建立目錄結構
mkdir -p image_processor/{include,src,data}

# 建立空檔案
touch image_processor/include/{image.h,filters.h,utils.h}
touch image_processor/src/{main.cu,filters.cu,image_io.cu}
touch image_processor/Makefile
```

### 基本 Makefile

```makefile
NVCC = nvcc
CFLAGS = -O3 -arch=sm_50

TARGET = image_processor
SRCS = src/main.cu src/filters.cu src/image_io.cu
INCLUDES = -Iinclude

all: $(TARGET)

$(TARGET): $(SRCS)
	$(NVCC) $(CFLAGS) $(INCLUDES) -o $@ $^

clean:
	rm -f $(TARGET)

.PHONY: all clean
```

---

**明天我們將開始實作核心功能！**
