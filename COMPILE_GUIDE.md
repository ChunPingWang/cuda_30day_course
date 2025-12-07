# CUDA 課程編譯指南

## 編譯環境需求

要編譯本課程的 CUDA 範例程式，需要以下環境：

### 方法一：安裝 Visual Studio Build Tools（推薦）

1. 下載 Visual Studio Installer：
   - 網址：https://visualstudio.microsoft.com/downloads/
   - 選擇 "Build Tools for Visual Studio 2022"

2. 安裝時選擇：
   - **"Desktop development with C++"** 工作負載
   - 確保包含 "MSVC v143 - VS 2022 C++ x64/x86 build tools"

3. 安裝完成後，使用 **"x64 Native Tools Command Prompt for VS 2022"** 來編譯

### 方法二：使用 Visual Studio Developer Command Prompt

如果已安裝 Visual Studio：

1. 開始選單搜尋 "Developer Command Prompt" 或 "x64 Native Tools Command Prompt"
2. 在該命令提示字元中執行 nvcc 命令

### 方法三：設定 PATH 環境變數

手動將 Visual Studio 的編譯器路徑加入 PATH：

```cmd
# 在 CMD 中執行（路徑可能因版本而異）
"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
```

## 編譯指令

### 基本編譯

```bash
cd week1/day1
nvcc device_query.cu -o device_query.exe
./device_query.exe
```

### 指定 GPU 架構

```bash
# 對於 RTX 4060（Ada Lovelace，SM 8.9）
nvcc -arch=sm_89 program.cu -o program.exe

# 對於 RTX 30 系列（Ampere，SM 8.6）
nvcc -arch=sm_86 program.cu -o program.exe

# 通用（讓 nvcc 自動選擇）
nvcc program.cu -o program.exe
```

### 編譯所有範例（批次腳本）

在 "x64 Native Tools Command Prompt" 中執行：

```cmd
@echo off
setlocal enabledelayedexpansion

for /r %%f in (*.cu) do (
    echo Compiling %%f...
    nvcc "%%f" -o "%%~dpnf.exe" 2>&1
    if !errorlevel! equ 0 (
        echo   Success!
    ) else (
        echo   Failed!
    )
)
```

## 執行範例

```bash
# Day 1: GPU 資訊查詢
cd week1/day1
./device_query.exe

# Day 2: Hello World
cd week1/day2
./hello_world.exe

# Day 4: 向量加法
cd week1/day4
./vector_add.exe
```

## 常見問題

### 1. "Cannot find compiler 'cl.exe' in PATH"

**原因**：nvcc 找不到 Visual Studio 的 C++ 編譯器

**解決方案**：
- 使用 "x64 Native Tools Command Prompt for VS" 執行
- 或執行 vcvars64.bat 設定環境

### 2. "nvcc: command not found"

**原因**：CUDA Toolkit 未正確加入 PATH

**解決方案**：
```cmd
set PATH=%PATH%;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin
```

### 3. 編譯警告 "Support for offline compilation..."

**原因**：nvcc 預設支援較舊的 GPU 架構

**解決方案**：
```bash
nvcc -arch=sm_89 -Wno-deprecated-gpu-targets program.cu -o program.exe
```

## 驗證安裝

```cmd
# 檢查 CUDA 版本
nvcc --version

# 檢查 GPU
nvidia-smi

# 檢查編譯器
cl
```

## 測試第一個程式

```cmd
cd "C:\Users\Rex Wang\workspace\cuda\cuda_30day_course\week1\day1"
nvcc device_query.cu -o device_query.exe
device_query.exe
```

應該會看到 GPU 資訊輸出，包括：
- GPU 名稱
- 計算能力
- 記憶體大小
- SM 數量
- 執行緒限制等
