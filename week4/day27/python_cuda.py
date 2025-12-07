"""
Day 27: Python 與 CUDA 整合範例

展示使用 CuPy, NumPy, 和 PyTorch 進行 GPU 計算
"""

import time
import numpy as np

# 嘗試導入 GPU 函式庫
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    print("CuPy 未安裝，跳過 CuPy 範例")

try:
    import torch
    HAS_TORCH = torch.cuda.is_available()
except ImportError:
    HAS_TORCH = False
    print("PyTorch 未安裝或無 CUDA 支援")


def benchmark_numpy(n):
    """使用 NumPy (CPU) 進行矩陣運算"""
    a = np.random.randn(n, n).astype(np.float32)
    b = np.random.randn(n, n).astype(np.float32)

    # 暖機
    _ = np.dot(a, b)

    # 計時
    start = time.time()
    for _ in range(10):
        c = np.dot(a, b)
    elapsed = (time.time() - start) / 10

    return elapsed * 1000  # 轉換為毫秒


def benchmark_cupy(n):
    """使用 CuPy (GPU) 進行矩陣運算"""
    if not HAS_CUPY:
        return None

    a = cp.random.randn(n, n).astype(cp.float32)
    b = cp.random.randn(n, n).astype(cp.float32)

    # 暖機
    _ = cp.dot(a, b)
    cp.cuda.Stream.null.synchronize()

    # 計時
    start = cp.cuda.Event()
    end = cp.cuda.Event()

    start.record()
    for _ in range(10):
        c = cp.dot(a, b)
    end.record()
    end.synchronize()

    elapsed = cp.cuda.get_elapsed_time(start, end) / 10

    return elapsed


def benchmark_pytorch(n):
    """使用 PyTorch (GPU) 進行矩陣運算"""
    if not HAS_TORCH:
        return None

    device = torch.device('cuda')

    a = torch.randn(n, n, device=device)
    b = torch.randn(n, n, device=device)

    # 暖機
    _ = torch.mm(a, b)
    torch.cuda.synchronize()

    # 計時
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(10):
        c = torch.mm(a, b)
    end.record()
    torch.cuda.synchronize()

    elapsed = start.elapsed_time(end) / 10

    return elapsed


def cupy_custom_kernel_example():
    """CuPy 自定義 CUDA 核心範例"""
    if not HAS_CUPY:
        return

    print("\n【CuPy 自定義 CUDA 核心】")

    # 定義自定義核心
    vector_add_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void vector_add(const float* a, const float* b, float* c, int n) {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx < n) {
            c[idx] = a[idx] + b[idx];
        }
    }
    ''', 'vector_add')

    n = 1000000
    a = cp.random.randn(n).astype(cp.float32)
    b = cp.random.randn(n).astype(cp.float32)
    c = cp.zeros(n, dtype=cp.float32)

    threads_per_block = 256
    blocks = (n + threads_per_block - 1) // threads_per_block

    # 執行自定義核心
    vector_add_kernel((blocks,), (threads_per_block,), (a, b, c, n))
    cp.cuda.Stream.null.synchronize()

    # 驗證
    expected = a + b
    print(f"自定義核心結果正確: {cp.allclose(c, expected)}")


def pytorch_example():
    """PyTorch GPU 範例"""
    if not HAS_TORCH:
        return

    print("\n【PyTorch GPU 範例】")

    device = torch.device('cuda')

    # 創建張量
    x = torch.randn(1000, 1000, device=device)

    # 各種運算
    print("執行 GPU 運算...")

    # 矩陣運算
    y = torch.mm(x, x.T)

    # 元素運算
    z = torch.relu(y)

    # 歸約
    mean = torch.mean(z)
    std = torch.std(z)

    print(f"結果形狀: {y.shape}")
    print(f"平均值: {mean.item():.4f}")
    print(f"標準差: {std.item():.4f}")

    # GPU 記憶體使用
    print(f"\nGPU 記憶體使用:")
    print(f"  已分配: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
    print(f"  快取:   {torch.cuda.memory_reserved() / 1024**2:.1f} MB")


def main():
    print("=" * 50)
    print("    Python GPU 計算效能比較")
    print("=" * 50)

    # GPU 資訊
    if HAS_CUPY:
        print(f"\nCuPy 版本: {cp.__version__}")
        print(f"CUDA 版本: {cp.cuda.runtime.runtimeGetVersion()}")

    if HAS_TORCH:
        print(f"\nPyTorch 版本: {torch.__version__}")
        print(f"CUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 效能比較
    print("\n【矩陣乘法效能比較】")
    sizes = [512, 1024, 2048, 4096]

    print(f"\n{'大小':>6} | {'NumPy (ms)':>12} | {'CuPy (ms)':>12} | {'PyTorch (ms)':>12} | {'加速比':>10}")
    print("-" * 65)

    for n in sizes:
        time_numpy = benchmark_numpy(n)
        time_cupy = benchmark_cupy(n)
        time_torch = benchmark_pytorch(n)

        # 計算加速比（相對於 NumPy）
        speedup = ""
        if time_cupy:
            speedup = f"{time_numpy / time_cupy:.1f}x (CuPy)"
        elif time_torch:
            speedup = f"{time_numpy / time_torch:.1f}x (Torch)"

        print(f"{n:>6} | {time_numpy:>12.2f} | "
              f"{time_cupy if time_cupy else 'N/A':>12} | "
              f"{time_torch if time_torch else 'N/A':>12} | "
              f"{speedup:>10}")

    # 進階範例
    cupy_custom_kernel_example()
    pytorch_example()

    print("\n" + "=" * 50)
    print("    總結")
    print("=" * 50)
    print("""
1. CuPy: NumPy 的 GPU 版本，API 幾乎相同
   - 適合科學計算和資料處理
   - 支援自定義 CUDA 核心

2. PyTorch: 深度學習框架
   - 自動微分支援
   - 廣泛的神經網路層

3. 對於大型資料，GPU 可以達到 10-100 倍加速

4. 小型資料可能因為傳輸開銷而更慢
""")


if __name__ == "__main__":
    main()
