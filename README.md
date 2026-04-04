# 30 天 CUDA 學習課程（高中生程度）

歡迎來到 CUDA 程式設計學習之旅！這是一個為期 30 天的完整課程，專為高中生設計。

## 課程目標

- 理解 GPU 平行運算的基本概念
- 學會使用 CUDA C/C++ 編寫 GPU 程式
- 掌握 CUDA 記憶體管理和優化技巧
- 能夠解決實際問題並優化效能

## 環境需求

- NVIDIA GPU（已驗證：RTX 4060 Laptop GPU, Compute Capability 8.9）
- NVIDIA GPU 驅動程式 571.96+
- CUDA Toolkit 12.8

支援兩種開發環境：

| 環境 | 編譯器 | 額外需求 |
|------|--------|----------|
| **Windows** | nvcc 12.8 | Visual Studio 2022+ Build Tools |
| **WSL (Ubuntu)** | nvcc 12.8 | `build-essential`, `cuda-toolkit-12-8` |

## 編譯與執行

### Windows

**環境設定：** 開啟 Developer Command Prompt for VS 2022，或手動載入環境變數：

**Command Prompt（cmd）：**

```cmd
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
```

**PowerShell：**

```powershell
& "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64
```

> 如果安裝的是 Build Tools 版本，將路徑中的 `Community` 替換為 `BuildTools`。

**編譯單一檔案：**

Command Prompt：

```cmd
nvcc -allow-unsupported-compiler -Wno-deprecated-gpu-targets -o hello_world.exe week1\day2\hello_world.cu
hello_world.exe
```

PowerShell：

```powershell
nvcc -allow-unsupported-compiler -Wno-deprecated-gpu-targets -o hello_world.exe week1/day2/hello_world.cu
.\hello_world.exe
```

**一次編譯所有檔案：**

Command Prompt：

```cmd
compile_all.bat
```

PowerShell：

```powershell
.\compile_all.bat
```

> `-allow-unsupported-compiler`：允許較新版本的 MSVC 編譯器。
> `-Wno-deprecated-gpu-targets`：隱藏舊架構棄用警告。

### WSL (Ubuntu)

**環境設定（首次）：**

```bash
# 安裝 CUDA Toolkit（不需要安裝驅動，驅動由 Windows 提供）
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y build-essential cmake cuda-toolkit-12-8

# 加入 PATH（加到 ~/.bashrc）
echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

**驗證安裝：**

```bash
nvidia-smi        # 確認 GPU 可見
nvcc --version    # 確認 CUDA 編譯器
```

**編譯單一檔案：**

```bash
nvcc -Wno-deprecated-gpu-targets -o hello_world week1/day2/hello_world.cu
./hello_world
```

**使用 Makefile 編譯：**

```bash
make              # 編譯所有範例
make week1        # 只編譯第一週
make week2        # 只編譯第二週
make clean        # 清除所有編譯產出
```

### 快速驗證（兩個平台皆適用）

```bash
# 1. 查詢 GPU 資訊
nvcc -Wno-deprecated-gpu-targets -o device_query week1/day1/device_query.cu
./device_query

# 2. Hello World
nvcc -Wno-deprecated-gpu-targets -o hello_world week1/day2/hello_world.cu
./hello_world

# 3. 向量加法
nvcc -Wno-deprecated-gpu-targets -o vector_add week1/day4/vector_add.cu
./vector_add
```

## 課程結構

### 第一週：CUDA 基礎與入門
- **Day 1**: CUDA 簡介與環境設定驗證
- **Day 2**: 第一個 CUDA 程式 - Hello World
- **Day 3**: 理解核心函數（Kernel）與執行緒
- **Day 4**: 陣列加法 - 第一個平行運算
- **Day 5**: CUDA 記憶體管理基礎
- **Day 6**: 執行緒索引與資料對應
- **Day 7**: 週末練習與複習

### 第二週：深入 CUDA 架構
- **Day 8**: Grid、Block 和 Thread 的層次結構
- **Day 9**: 向量運算與最佳化
- **Day 10**: 矩陣運算入門
- **Day 11**: 共享記憶體（Shared Memory）
- **Day 12**: 同步與協調（Synchronization）
- **Day 13**: 錯誤處理與除錯技巧
- **Day 14**: 週末專題：矩陣乘法

### 第三週：進階技巧與優化
- **Day 15**: 記憶體合併（Memory Coalescing）
- **Day 16**: 紋理記憶體與常數記憶體
- **Day 17**: 掃描（Scan/Prefix Sum）演算法
- **Day 18**: 平行歸約（Reduction）
- **Day 19**: 直方圖計算
- **Day 20**: 排序演算法（Bitonic Sort）
- **Day 21**: 週末專題：圖像處理

### 第四週：實戰應用與專題
- **Day 22**: 進階圖像處理
- **Day 23**: CUDA Streams 與非同步執行
- **Day 24**: 多 GPU 程式設計
- **Day 25**: 動態平行處理（Dynamic Parallelism）
- **Day 26**: Unified Memory 進階使用
- **Day 27**: Python 與 CUDA 整合（CuPy/PyTorch）
- **Day 28**: 期末專題（一）- 規劃與架構
- **Day 29**: 期末專題（二）- 核心實作
- **Day 30**: 課程總結與未來學習方向

## 範例程式清單

| 週 | 天 | 檔案 | 說明 |
|----|-----|------|------|
| 1 | Day 1 | `device_query.cu` | GPU 裝置資訊查詢 |
| 1 | Day 2 | `hello_world.cu`, `hello_advanced.cu` | 第一個 CUDA 程式 |
| 1 | Day 3 | `thread_index.cu`, `array_index.cu` | 執行緒索引 |
| 1 | Day 4 | `vector_add.cu`, `vector_add_benchmark.cu` | 向量加法與效能測試 |
| 1 | Day 5 | `memory_basics.cu`, `unified_memory.cu` | 記憶體管理 |
| 1 | Day 6 | `2d_indexing.cu`, `stride_pattern.cu` | 2D 索引與 Stride |
| 1 | Day 7 | `ex1_square.cu` ~ `ex4_find_max.cu` | 週末練習 |
| 2 | Day 8 | `warp_info.cu`, `divergence_demo.cu` | Warp 與分支分歧 |
| 2 | Day 9 | `vector_add_optimized.cu` | 向量加法優化版 |
| 2 | Day 10 | `matrix_mul_basic.cu` | 基礎矩陣乘法 |
| 2 | Day 11 | `matrix_mul_tiled.cu` | Tiled 矩陣乘法 |
| 2 | Day 12 | `atomic_ops.cu` | 原子操作 |
| 2 | Day 13 | `error_handling.cu` | 錯誤處理 |
| 2 | Day 14 | `matrix_mul_complete.cu` | 完整矩陣乘法 |
| 3 | Day 15 | `memory_coalescing.cu` | 記憶體合併 |
| 3 | Day 16 | `constant_memory.cu` | 常數記憶體 |
| 3 | Day 17 | `scan_basic.cu` | Prefix Sum |
| 3 | Day 18 | `reduction_basic.cu` | 平行歸約 |
| 3 | Day 19 | `histogram.cu` | 直方圖 |
| 3 | Day 20 | `bitonic_sort.cu` | Bitonic 排序 |
| 3 | Day 21 | `image_processing.cu` | 圖像處理 |
| 4 | Day 22 | `image_filters.cu` | 進階圖像濾鏡 |
| 4 | Day 23 | `streams.cu` | CUDA Streams |
| 4 | Day 24 | `multi_gpu.cu` | 多 GPU |
| 4 | Day 26 | `unified_memory_advanced.cu` | Unified Memory 進階 |

## 學習建議

1. **每天預留 1-2 小時**學習時間
2. **動手實作**每個範例程式碼
3. **完成每日練習**鞏固學習成果
4. **遇到問題**先嘗試自己解決，再查閱文檔
5. **週末複習**本週所學內容

## 常見問題

**Q: WSL 裡 `nvidia-smi` 看得到 GPU 但程式偵測不到？**
A: 確認安裝的 CUDA Toolkit 版本不超過驅動支援的版本。用 `nvidia-smi` 右上角顯示的 CUDA Version 為上限。

**Q: Windows 編譯出現 `cl.exe not found`？**
A: 需先載入 Visual Studio 環境，使用 Developer Command Prompt 或執行 `vcvarsall.bat x64`。

**Q: WSL 需要安裝 NVIDIA 驅動嗎？**
A: 不需要。WSL 的 GPU 驅動由 Windows 端提供，只需在 WSL 內安裝 `cuda-toolkit`。
