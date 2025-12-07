# 30 天 CUDA 學習課程（高中生程度）

歡迎來到 CUDA 程式設計學習之旅！這是一個為期 30 天的完整課程，專為高中生設計。

## 課程目標

- 理解 GPU 平行運算的基本概念
- 學會使用 CUDA C/C++ 編寫 GPU 程式
- 掌握 CUDA 記憶體管理和優化技巧
- 能夠解決實際問題並優化效能

## 課程結構

### 📚 第一週：CUDA 基礎與入門
- **Day 1**: CUDA 簡介與環境設定驗證
- **Day 2**: 第一個 CUDA 程式 - Hello World
- **Day 3**: 理解核心函數（Kernel）與執行緒
- **Day 4**: 陣列加法 - 第一個平行運算
- **Day 5**: CUDA 記憶體管理基礎
- **Day 6**: 執行緒索引與資料對應
- **Day 7**: 週末練習與複習

### 🚀 第二週：深入 CUDA 架構
- **Day 8**: Grid、Block 和 Thread 的層次結構
- **Day 9**: 向量運算與最佳化
- **Day 10**: 矩陣運算入門
- **Day 11**: 共享記憶體（Shared Memory）
- **Day 12**: 同步與協調（Synchronization）
- **Day 13**: 錯誤處理與除錯技巧
- **Day 14**: 週末專題：矩陣乘法

### 💡 第三週：進階技巧與優化
- **Day 15**: 記憶體合併（Memory Coalescing）
- **Day 16**: 紋理記憶體與常數記憶體
- **Day 17**: 掃描（Scan/Prefix Sum）演算法
- **Day 18**: 平行歸約（Reduction）
- **Day 19**: 直方圖計算
- **Day 20**: 排序演算法（Bitonic Sort）
- **Day 21**: 週末專題：圖像處理

### 🎯 第四週：實戰應用與專題
- **Day 22**: 進階圖像處理
- **Day 23**: CUDA Streams 與非同步執行
- **Day 24**: 多 GPU 程式設計
- **Day 25**: 動態平行處理（Dynamic Parallelism）
- **Day 26**: Unified Memory 進階使用
- **Day 27**: Python 與 CUDA 整合（CuPy/PyTorch）
- **Day 28**: 期末專題（一）- 規劃與架構
- **Day 29**: 期末專題（二）- 核心實作
- **Day 30**: 課程總結與未來學習方向

## 學習建議

1. **每天預留 1-2 小時**學習時間
2. **動手實作**每個範例程式碼
3. **完成每日練習**鞏固學習成果
4. **遇到問題**先嘗試自己解決，再查閱文檔
5. **週末複習**本週所學內容

## 環境需求

- ✅ NVIDIA GPU（已驗證：RTX 4060 Laptop GPU）
- ✅ CUDA Toolkit 12.8
- ✅ Python 3.12 + PyTorch（選修）
- ✅ 編譯器：nvcc

## 如何使用本課程

1. 按照天數順序學習
2. 每天閱讀教學文檔（README.md）
3. 執行並理解範例程式碼
4. 完成當天的練習題
5. 週末進行總複習

## 快速開始

進入第一週第一天：
```bash
cd week1/day1
cat README.md
```

編譯並執行範例：
```bash
nvcc example.cu -o example
./example
```

---

**準備好了嗎？讓我們開始這段激動人心的 GPU 程式設計之旅吧！** 🚀
