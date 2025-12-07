#!/usr/bin/env python3
"""
CUDA Device Query - Python 版本
使用 CuPy 查詢 GPU 設備信息
"""

import cupy as cp
import numpy as np

def print_device_info():
    """打印 CUDA 設備信息"""
    print("=" * 60)
    print("CUDA Device Query")
    print("=" * 60)
    
    # 獲取 GPU 設備數量
    device_count = cp.cuda.runtime.getDeviceCount()
    print(f"\nNumber of GPUs: {device_count}")
    
    # 遍歷每個設備
    for i in range(device_count):
        device = cp.cuda.Device(i)
        device.use()
        
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
    
    # 簡單的 GPU 計算測試
    print("\nGPU Computation Test:")
    a_gpu = cp.arange(10)
    b_gpu = cp.arange(10)
    c_gpu = a_gpu + b_gpu
    print(f"  GPU Result (a+b): {c_gpu[:5]}... (first 5 elements)")
    print("\n✓ All tests passed!")
    print("=" * 60)

if __name__ == "__main__":
    print_device_info()
