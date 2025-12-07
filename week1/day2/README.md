# Day 2: 第一個 CUDA 程式 - Hello World

## 📚 今日學習目標

- 了解 CUDA 程式的基本結構
- 學習什麼是「核心函數」（Kernel）
- 寫出並執行第一個 CUDA 程式
- 理解主機（Host）和設備（Device）的概念

## 🌍 CUDA 程式的世界觀

在 CUDA 程式中，有兩個重要的角色：

- **主機（Host）**：你的 CPU 和系統記憶體（RAM）
- **設備（Device）**：你的 GPU 和顯示記憶體（VRAM）

就像是：
- **主機 = 指揮官**：負責發號施令、準備資料
- **設備 = 千軍萬馬**：負責執行大量的平行任務

## 🔑 CUDA 程式的基本結構

一個典型的 CUDA 程式包含：

1. **主機程式碼**：普通的 C/C++ 程式碼
2. **設備程式碼**：在 GPU 上執行的程式碼（核心函數）
3. **記憶體管理**：在主機和設備之間傳輸資料

### 核心函數（Kernel）

核心函數是在 GPU 上執行的特殊函數，用 `__global__` 關鍵字標記：

```cuda
__global__ void myKernel() {
    // 這段程式碼會在 GPU 上執行！
}
```

## 💻 Hello World from GPU！

讓我們來看第一個 CUDA 程式（`hello_world.cu`）：

### 程式碼解析

```cuda
#include <stdio.h>

// 這是一個核心函數！
__global__ void helloFromGPU() {
    printf("Hello World from GPU!\n");
}

int main() {
    // 從 CPU 打招呼
    printf("Hello World from CPU!\n");

    // 啟動核心函數：1 個 Block，1 個 Thread
    helloFromGPU<<<1, 1>>>();

    // 等待 GPU 完成工作
    cudaDeviceSynchronize();

    return 0;
}
```

### 重點說明

#### 1. `__global__` 關鍵字
- 告訴編譯器：「這個函數要在 GPU 上執行」
- 必須從主機（CPU）呼叫
- 返回值必須是 `void`

#### 2. 核心函數啟動語法
```cuda
helloFromGPU<<<1, 1>>>();
         啟動配置
```

`<<<1, 1>>>` 是 CUDA 的特殊語法：
- 第一個 `1`：使用 1 個 Block
- 第二個 `1`：每個 Block 有 1 個 Thread

（明天我們會詳細學習這個！）

#### 3. `cudaDeviceSynchronize()`
- CPU 和 GPU 是異步執行的
- 這個函數讓 CPU 等待 GPU 完成工作
- 就像老師說「等大家都做完再下課」

## 🚀 編譯與執行

### 步驟 1：編譯
```bash
nvcc hello_world.cu -o hello_world
```

`nvcc` 是 NVIDIA 的 CUDA 編譯器。

### 步驟 2：執行
```bash
./hello_world
```

### 預期輸出
```
Hello World from CPU!
Hello World from GPU!
```

## 🔬 實驗時間

試試修改程式：

### 實驗 1：啟動多個執行緒
把 `<<<1, 1>>>` 改成 `<<<1, 10>>>`：
```cuda
helloFromGPU<<<1, 10>>>();
```

會發生什麼事？（提示：你會看到 10 次 "Hello World from GPU!"）

### 實驗 2：使用多個 Block
試試 `<<<5, 1>>>` 或 `<<<2, 5>>>`：
```cuda
helloFromGPU<<<5, 1>>>();
```

觀察輸出的次數！

## 📝 今日作業

1. ✅ 編譯並執行 `hello_world.cu`
2. ✅ 完成實驗 1 和實驗 2
3. ✅ 執行 `hello_advanced.cu`（增強版）
4. ✅ 回答思考問題

## 🎯 練習題

創建一個新程式 `greeting.cu`，讓 GPU 輸出：
```
GPU says: Welcome to CUDA Programming!
```

提示：只需要修改 `printf` 的內容！

## 🤓 重要概念總結

| 術語 | 說明 |
|------|------|
| Host | CPU 和系統記憶體 |
| Device | GPU 和顯示記憶體 |
| Kernel | 在 GPU 上執行的函數 |
| `__global__` | 核心函數的標記 |
| `<<<>>>` | 核心函數啟動配置 |
| `cudaDeviceSynchronize()` | 等待 GPU 完成 |

## ❓ 思考問題

1. 為什麼需要 `cudaDeviceSynchronize()`？如果不加會怎樣？
2. CPU 和 GPU 的 `printf` 有什麼不同？
3. 當我們使用 `<<<5, 10>>>` 時，總共會執行多少次核心函數？

## 🎁 彩蛋

你知道嗎？GPU 的執行緒可以達到數千到數百萬個！想像一下 `<<<1024, 1024>>>` 會發生什麼... 😱

---

**明天我們將深入學習執行緒和索引，這是 CUDA 的核心概念！** 🎯
