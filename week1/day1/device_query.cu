#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    printf("========================================\n");
    printf("    CUDA 設備資訊查詢程式\n");
    printf("========================================\n\n");

    // 取得 CUDA 設備數量
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    printf("系統中的 CUDA 設備數量: %d\n\n", deviceCount);

    // 如果沒有 CUDA 設備
    if (deviceCount == 0) {
        printf("錯誤：沒有找到支援 CUDA 的設備！\n");
        return 1;
    }

    // 遍歷每個 CUDA 設備
    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        printf("設備 %d: %s\n", dev, deviceProp.name);
        printf("----------------------------------------\n");

        // 計算能力
        printf("  計算能力: %d.%d\n",
               deviceProp.major, deviceProp.minor);

        // CUDA 核心數（簡化計算）
        int cores = 0;
        int mp = deviceProp.multiProcessorCount;

        // 根據計算能力估算核心數
        if (deviceProp.major == 8) {
            cores = mp * 128; // Ampere 架構
        } else if (deviceProp.major == 7) {
            cores = mp * 64;  // Turing/Volta 架構
        } else if (deviceProp.major == 6) {
            cores = mp * 64;  // Pascal 架構
        } else if (deviceProp.major == 9) {
            cores = mp * 128; // Hopper 架構
        }

        printf("  串流多處理器 (SM) 數量: %d\n", mp);
        printf("  估計 CUDA 核心數: %d\n", cores);

        // 記憶體資訊
        printf("  全域記憶體大小: %.2f GB\n",
               deviceProp.totalGlobalMem / 1024.0 / 1024.0 / 1024.0);
        printf("  每個 Block 共享記憶體: %.2f KB\n",
               deviceProp.sharedMemPerBlock / 1024.0);
        printf("  常數記憶體大小: %.2f KB\n",
               deviceProp.totalConstMem / 1024.0);

        // 執行緒資訊
        printf("  每個 Block 最大執行緒數: %d\n",
               deviceProp.maxThreadsPerBlock);
        printf("  每個 SM 最大執行緒數: %d\n",
               deviceProp.maxThreadsPerMultiProcessor);

        // Block 維度
        printf("  Block 最大維度: (%d, %d, %d)\n",
               deviceProp.maxThreadsDim[0],
               deviceProp.maxThreadsDim[1],
               deviceProp.maxThreadsDim[2]);

        // Grid 維度
        printf("  Grid 最大維度: (%d, %d, %d)\n",
               deviceProp.maxGridSize[0],
               deviceProp.maxGridSize[1],
               deviceProp.maxGridSize[2]);

        // 時脈頻率
        printf("  時脈頻率: %.2f GHz\n",
               deviceProp.clockRate / 1e6);

        // Warp 大小
        printf("  Warp 大小: %d\n", deviceProp.warpSize);

        printf("\n");
    }

    printf("========================================\n");
    printf("CUDA 環境驗證完成！\n");
    printf("========================================\n");

    return 0;
}
