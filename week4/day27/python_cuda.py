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
    """使用 NumPy (CPU) 進行矩陣運算，作為 GPU 效能的基準對照"""
    # np.float32 = 單精度浮點數，和 GPU 常用的精度一致，確保公平比較
    a = np.random.randn(n, n).astype(np.float32)
    b = np.random.randn(n, n).astype(np.float32)

    # 暖機：第一次執行通常較慢（快取未建立），所以先跑一次不計時
    _ = np.dot(a, b)

    # 計時：跑 10 次取平均，減少隨機誤差
    start = time.time()
    for _ in range(10):
        c = np.dot(a, b)
    elapsed = (time.time() - start) / 10

    return elapsed * 1000  # 轉換為毫秒


def benchmark_cupy(n):
    """使用 CuPy (GPU) 進行矩陣運算 — CuPy 的 API 幾乎和 NumPy 一模一樣"""
    if not HAS_CUPY:
        return None

    # cp.random.randn 直接在 GPU 上產生隨機數（資料從頭到尾都在 GPU 上）
    a = cp.random.randn(n, n).astype(cp.float32)
    b = cp.random.randn(n, n).astype(cp.float32)

    # 暖機
    _ = cp.dot(a, b)
    # ⚠️ 注意：GPU 操作是非同步的，synchronize() 確保計算真正完成
    cp.cuda.Stream.null.synchronize()

    # 使用 CUDA Event 計時（比 Python 的 time.time() 更精確，直接測量 GPU 時間）
    start = cp.cuda.Event()
    end = cp.cuda.Event()

    start.record()
    for _ in range(10):
        c = cp.dot(a, b)
    end.record()
    end.synchronize()  # 等待所有 GPU 計算完成

    elapsed = cp.cuda.get_elapsed_time(start, end) / 10

    return elapsed


def benchmark_pytorch(n):
    """使用 PyTorch (GPU) 進行矩陣運算 — PyTorch 是最熱門的深度學習框架"""
    if not HAS_TORCH:
        return None

    # 💡 Debug 提示：如果出現 "CUDA out of memory"，試著把矩陣大小 n 調小
    device = torch.device('cuda')  # 指定使用 GPU

    # device=device 讓張量直接在 GPU 上建立
    a = torch.randn(n, n, device=device)
    b = torch.randn(n, n, device=device)

    # 暖機
    _ = torch.mm(a, b)  # torch.mm = 矩陣乘法
    torch.cuda.synchronize()  # 等 GPU 完成（因為 PyTorch 的 GPU 操作是非同步的）

    # 使用 CUDA Event 計時
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(10):
        c = torch.mm(a, b)
    end.record()
    # ⚠️ 注意：一定要 synchronize，否則 elapsed_time 會得到錯誤的數值
    torch.cuda.synchronize()

    elapsed = start.elapsed_time(end) / 10

    return elapsed


def cupy_custom_kernel_example():
    """CuPy 自定義 CUDA 核心範例"""
    if not HAS_CUPY:
        return

    print("\n【CuPy 自定義 CUDA 核心】")

    # cp.RawKernel 可以直接在 Python 裡寫 CUDA C 程式碼
    # 第一個參數是 CUDA 原始碼（字串），第二個參數是 kernel 函式名稱
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

    # grid 和 block 的設定方式和 CUDA C 一樣
    threads_per_block = 256
    blocks = (n + threads_per_block - 1) // threads_per_block  # 無條件進位除法

    # 執行自定義核心：(grid_size,), (block_size,), (參數們)
    # ⚠️ 注意：grid 和 block 大小必須是 tuple，即使只有一維也要加逗號
    vector_add_kernel((blocks,), (threads_per_block,), (a, b, c, n))
    cp.cuda.Stream.null.synchronize()  # 等待 GPU 完成

    # 驗證
    expected = a + b
    print(f"自定義核心結果正確: {cp.allclose(c, expected)}")


def pytorch_example():
    """PyTorch GPU 範例"""
    if not HAS_TORCH:
        return

    print("\n【PyTorch GPU 範例】")

    device = torch.device('cuda')

    # 創建張量（tensor = PyTorch 的多維陣列，類似 NumPy 的 ndarray）
    x = torch.randn(1000, 1000, device=device)

    # 各種運算（全部在 GPU 上執行）
    print("執行 GPU 運算...")

    y = torch.mm(x, x.T)  # 矩陣乘法，x.T 是轉置
    z = torch.relu(y)      # ReLU 激活函數：把負數變成 0（深度學習常用）

    # 歸約運算：把整個張量縮減成一個數值
    mean = torch.mean(z)
    std = torch.std(z)

    print(f"結果形狀: {y.shape}")
    print(f"平均值: {mean.item():.4f}")
    print(f"標準差: {std.item():.4f}")

    # GPU 記憶體使用（用來檢查是否有記憶體洩漏）
    # 💡 Debug 提示：如果 memory_allocated 持續增加，可能有張量沒被釋放
    print(f"\nGPU 記憶體使用:")
    print(f"  已分配: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")  # 目前使用中的記憶體
    print(f"  快取:   {torch.cuda.memory_reserved() / 1024**2:.1f} MB")   # PyTorch 快取池的大小


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
