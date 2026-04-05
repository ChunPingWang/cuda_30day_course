# Day 1: CUDA 簡介與環境設定驗證

## 📚 今日學習目標

- 了解什麼是 GPU 和 CUDA
- 理解為什麼需要 GPU 運算
- 驗證你的 CUDA 環境是否正確安裝
- 運行第一個簡單的 CUDA 程式

## 🤔 什麼是 GPU？

**GPU（Graphics Processing Unit，圖形處理器）** 原本是為了處理電腦遊戲和圖形而設計的。但科學家發現，GPU 非常擅長同時處理大量相似的計算任務。

### CPU vs GPU 的比喻

想像你要在圖書館整理 10,000 本書：

- **CPU（中央處理器）**：像是一個非常聰明、速度很快的圖書管理員，但只有 1 個人（或 4-8 個人）。他可以做各種複雜的工作，但一次只能處理少數幾本書。

- **GPU（圖形處理器）**：像是有 1000 個助手的團隊，每個助手雖然沒有管理員聰明，但他們可以同時整理 1000 本書！對於簡單重複的工作，GPU 快得多。

## 💡 什麼是 CUDA？

**CUDA（Compute Unified Device Architecture）** 是 NVIDIA 開發的平台，讓我們可以用類似 C 語言的方式來寫 GPU 程式。

就像你需要學英文才能跟外國人溝通，你需要學 CUDA 才能「指揮」GPU 工作。

## ⚡ 為什麼要學 CUDA？

CUDA 可以讓程式執行速度提升 **10 倍到 100 倍**！適用於：

- 🎮 遊戲開發
- 🤖 人工智慧/機器學習
- 🔬 科學計算
- 🎬 影片處理
- 💰 金融分析
- 🧬 生物資訊學

## 🔧 驗證你的環境

讓我們確認你的電腦已經準備好學習 CUDA！

### 步驟 1：檢查 CUDA 編譯器

打開終端機（命令提示字元或 Git Bash），輸入：

```bash
nvcc --version
```

你應該會看到類似這樣的輸出：
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Wed_Jan_15_19:38:46_Pacific_Standard_Time_2025
Cuda compilation tools, release 12.8, V12.8.61
```

### 步驟 2：檢查你的 GPU

```bash
nvidia-smi
```

你應該會看到你的 GPU 資訊（例如：RTX 4060 Laptop GPU）。

## 📝 第一個驗證程式

讓我們運行一個簡單的程式來確認一切正常！

查看並執行 `device_query.cu` 這個檔案，這個程式會顯示你的 GPU 的詳細資訊。

## 🔧 編譯與執行

### CUDA 編譯

#### device_query.cu - GPU 裝置資訊查詢

**Windows（cmd）：**

```cmd
nvcc -allow-unsupported-compiler -Wno-deprecated-gpu-targets -Xcompiler "/wd4819" -o device_query.exe device_query.cu
device_query.exe
```

**Windows（PowerShell）：**

```powershell
nvcc -allow-unsupported-compiler -Wno-deprecated-gpu-targets -Xcompiler "/wd4819" -o device_query.exe device_query.cu
.\device_query.exe
```

**WSL / Linux：**

```bash
nvcc -Wno-deprecated-gpu-targets -o device_query device_query.cu
./device_query
```

### Python 等效

已提供 `device_query.py`，可直接執行：

```bash
python device_query.py
```

也可以使用 CuPy 查詢裝置資訊：

```python
import cupy as cp
dev = cp.cuda.Device(0)
print(f"GPU: {dev.attributes}")
print(f"記憶體: {dev.mem_info}")
```

## 🎯 今日作業

1. ✅ 成功執行 `nvcc --version`
2. ✅ 成功執行 `nvidia-smi`
3. ✅ 編譯並執行 `device_query.cu`
4. ✅ 截圖或記錄你的 GPU 資訊

## 🤓 延伸閱讀

- GPU 有多少個核心？（你的 RTX 4060 有 3072 個 CUDA 核心！）
- GPU 記憶體有多大？
- 什麼是 CUDA Compute Capability？

## ❓ 思考問題

1. 為什麼 GPU 適合平行運算？
2. 什麼樣的問題適合用 GPU 解決？
3. 什麼樣的問題不適合用 GPU？

---

**明天我們將寫第一個真正的 CUDA 程式！** 🚀
