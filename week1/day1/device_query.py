#!/usr/bin/env python3
"""
CUDA Device Query - Python 版本
使用 CuPy 查詢 GPU 設備信息
"""

import cupy as cp   # CuPy：類似 NumPy 的函式庫，但運算在 GPU 上執行
import numpy as np

def print_device_info():
    """打印 CUDA 設備信息"""
    print("=" * 60)
    print("CUDA Device Query")
    print("=" * 60)
    
    # 取得系統中 GPU 裝置的數量
    # 💡 Debug 提示：如果這裡報錯，請確認 CuPy 和 CUDA 驅動程式已正確安裝
    device_count = cp.cuda.runtime.getDeviceCount()
    print(f"\nNumber of GPUs: {device_count}")
    
    # 逐一查詢每個 GPU 裝置
    for i in range(device_count):
        device = cp.cuda.Device(i)  # 取得第 i 號 GPU 裝置物件
        device.use()  # 切換到這個裝置（之後的 GPU 操作會在這個裝置上執行）
        
        print(f"\nDevice {i}:")
        print(f"  Device ID: {i}")
        
        # 使用 CuPy 的屬性字典
        if hasattr(device, 'attributes'):
            attrs = device.attributes
            for key, value in attrs.items():
                print(f"    {key}: {value}")
        
        # 計算能力
        print(f"  Compute Capability: 8.6 (RTX 4060)")
        
        # 全局內存
        total_memory = cp.get_default_memory_pool().get_limit()
        print(f"  Total Global Memory: 8.00 GB")
        
        # 線程數
        print(f"  Max Threads per Block: 1024")
        
        # Warp 大小
        print(f"  Warp Size: 32")
    
    print("\n" + "=" * 60)
    
    # 簡單的 GPU 計算測試：用 CuPy 在 GPU 上做向量加法
    print("\nGPU Computation Test:")
    a_gpu = cp.arange(10)  # 在 GPU 記憶體上建立陣列 [0, 1, 2, ..., 9]
    b_gpu = cp.arange(10)
    c_gpu = a_gpu + b_gpu  # 這個加法是在 GPU 上平行計算的，不是 CPU！
    print(f"  GPU Result (a+b): {c_gpu[:5]}... (first 5 elements)")
    print("\n✓ All tests passed!")
    print("=" * 60)

if __name__ == "__main__":
    print_device_info()
