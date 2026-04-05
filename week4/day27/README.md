# Day 27: CUDA 與 Python（PyCUDA/CuPy）

## 📚 今日學習目標

- 學習在 Python 中使用 CUDA
- 了解 PyCUDA 和 CuPy 的區別
- 整合 PyTorch 使用 CUDA
- 實作混合 Python/CUDA 應用

## 🐍 為什麼在 Python 中使用 CUDA？

- **快速原型開發**：Python 語法簡潔
- **豐富的生態系**：NumPy、Pandas、Matplotlib
- **機器學習整合**：PyTorch、TensorFlow
- **生產力高**：減少樣板程式碼

## 📊 選項比較

| 工具 | 特點 | 適用場景 |
|------|------|----------|
| **CuPy** | NumPy 語法，自動優化 | 快速開發，矩陣運算 |
| **PyCUDA** | 完整控制，自訂核心 | 需要自訂 CUDA 核心 |
| **Numba** | JIT 編譯 Python 函數 | 加速現有 Python 程式 |
| **PyTorch** | 深度學習框架 | 機器學習應用 |

## 🚀 CuPy 快速入門

### 安裝

```bash
pip install cupy-cuda12x  # 根據你的 CUDA 版本
```

### 基本使用

```python
import cupy as cp
import numpy as np

# 創建 GPU 陣列
a_gpu = cp.array([1, 2, 3, 4, 5])
b_gpu = cp.array([5, 4, 3, 2, 1])

# GPU 運算（語法與 NumPy 相同！）
c_gpu = a_gpu + b_gpu
print(c_gpu)  # [6 6 6 6 6]

# 轉回 NumPy
c_cpu = cp.asnumpy(c_gpu)
```

### 矩陣運算

```python
import cupy as cp
import time

# 創建大矩陣
n = 4096
a = cp.random.rand(n, n).astype(cp.float32)
b = cp.random.rand(n, n).astype(cp.float32)

# 矩陣乘法（自動使用 cuBLAS）
start = time.time()
c = cp.matmul(a, b)
cp.cuda.Stream.null.synchronize()
print(f"GPU 時間: {time.time() - start:.4f}s")
```

### 自訂核心函數

```python
import cupy as cp

# 定義 CUDA 核心
kernel_code = '''
extern "C" __global__ void add_kernel(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
'''

# 編譯核心
add_kernel = cp.RawKernel(kernel_code, 'add_kernel')

# 準備資料
n = 1000
a = cp.random.rand(n).astype(cp.float32)
b = cp.random.rand(n).astype(cp.float32)
c = cp.zeros(n, dtype=cp.float32)

# 執行核心
threads = 256
blocks = (n + threads - 1) // threads
add_kernel((blocks,), (threads,), (a, b, c, n))

print(c[:10])
```

## 🔥 PyTorch CUDA 操作

### 基本使用

```python
import torch

# 檢查 CUDA 可用性
print(f"CUDA 可用: {torch.cuda.is_available()}")
print(f"GPU 名稱: {torch.cuda.get_device_name(0)}")

# 創建 GPU Tensor
device = torch.device("cuda")
x = torch.rand(3, 3, device=device)
y = torch.rand(3, 3, device=device)

# GPU 運算
z = x + y
print(z)

# 移動 Tensor
x_cpu = x.cpu()
x_gpu = x_cpu.cuda()
```

### 自訂 CUDA 擴展

```python
import torch
from torch.utils.cpp_extension import load

# 動態編譯 CUDA 擴展
cuda_module = load(
    name='custom_ops',
    sources=['custom_ops.cu'],
    verbose=True
)

# 使用自訂操作
result = cuda_module.my_cuda_function(input_tensor)
```

## 🔧 實作練習

### 範例程式

1. **cupy_basics.py** - CuPy 基礎操作
2. **pytorch_cuda.py** - PyTorch CUDA 操作
3. **hybrid_example.py** - 混合 Python/CUDA 應用

## 🔧 編譯與執行

本日為 Python 整合課程，無 CUDA 程式需要編譯。

### 執行 Python 範例

```bash
python python_cuda.py
```

### CuPy 快速範例

```python
import cupy as cp
import numpy as np

# GPU 陣列運算
a = cp.random.rand(1000, 1000, dtype=cp.float32)
b = cp.random.rand(1000, 1000, dtype=cp.float32)
c = cp.matmul(a, b)

# 與 NumPy 互操作
result_np = cp.asnumpy(c)
print(f"結果形狀: {result_np.shape}, 總和: {result_np.sum():.2f}")
```

## 📝 今日作業

1. ✅ 安裝 CuPy
2. ✅ 使用 CuPy 實作矩陣乘法
3. ✅ 比較 NumPy 和 CuPy 的效能

---

**最後三天是期末專題製作！** 🎓
