#include <stdio.h>
#include <cuda_runtime.h> // CUDA 執行時期函式庫，提供 cudaGetDeviceCount 等 API

// 此程式用於查詢系統中所有 CUDA GPU 的硬體資訊（記憶體、核心數、執行緒限制等）
int main() {
    printf("========================================\n");
    printf("    CUDA Device Information Query\n");
    printf("========================================\n\n");

    // 取得系統中 CUDA 裝置（GPU）的數量
    int deviceCount;
    cudaGetDeviceCount(&deviceCount); // 💡 Debug 提示：如果回傳 0，表示沒有偵測到 GPU，請確認驅動程式是否已安裝

    printf("Number of CUDA devices: %d\n\n", deviceCount);

    // ⚠️ 注意：如果沒有找到 CUDA 裝置，程式會提前結束
    if (deviceCount == 0) {
        printf("Error: No CUDA-capable device found!\n");
        return 1;
    }

    // 逐一查詢每個 CUDA 裝置的詳細屬性
    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp deviceProp; // 用來儲存裝置屬性的結構體
        cudaGetDeviceProperties(&deviceProp, dev); // 將第 dev 號裝置的屬性填入 deviceProp

        printf("Device %d: %s\n", dev, deviceProp.name);
        printf("----------------------------------------\n");

        // 計算能力（Compute Capability）：決定 GPU 支援哪些 CUDA 功能
        printf("  Compute Capability: %d.%d\n",
               deviceProp.major, deviceProp.minor);

        // CUDA 核心數（簡化估算）：每個 SM 的核心數依架構不同而異
        int cores = 0;
        int mp = deviceProp.multiProcessorCount; // SM（串流多處理器）數量

        // 根據計算能力的主版本號來估算每個 SM 的核心數
        if (deviceProp.major == 8) {
            cores = mp * 128; // Ampere architecture
        } else if (deviceProp.major == 7) {
            cores = mp * 64;  // Turing/Volta architecture
        } else if (deviceProp.major == 6) {
            cores = mp * 64;  // Pascal architecture
        } else if (deviceProp.major == 9) {
            cores = mp * 128; // Hopper architecture
        }

        printf("  Streaming Multiprocessors (SM): %d\n", mp);
        printf("  Estimated CUDA Cores: %d\n", cores);

        // 記憶體資訊（GPU 有自己獨立的記憶體，和 CPU 的 RAM 是分開的）
        printf("  Global Memory: %.2f GB\n",
               deviceProp.totalGlobalMem / 1024.0 / 1024.0 / 1024.0);
        printf("  Shared Memory per Block: %.2f KB\n",
               deviceProp.sharedMemPerBlock / 1024.0);
        printf("  Constant Memory: %.2f KB\n",
               deviceProp.totalConstMem / 1024.0);

        // 執行緒（Thread）相關限制：寫 kernel 時要注意不要超過這些上限
        printf("  Max Threads per Block: %d\n",
               deviceProp.maxThreadsPerBlock);
        printf("  Max Threads per SM: %d\n",
               deviceProp.maxThreadsPerMultiProcessor);

        // Block 最大維度：Block 可以是 1D、2D 或 3D 的
        printf("  Max Block Dimensions: (%d, %d, %d)\n",
               deviceProp.maxThreadsDim[0],
               deviceProp.maxThreadsDim[1],
               deviceProp.maxThreadsDim[2]);

        // Grid 最大維度：Grid 是由多個 Block 組成的
        printf("  Max Grid Dimensions: (%d, %d, %d)\n",
               deviceProp.maxGridSize[0],
               deviceProp.maxGridSize[1],
               deviceProp.maxGridSize[2]);

        // Clock rate
        printf("  Clock Rate: %.2f GHz\n",
               deviceProp.clockRate / 1e6);

        // Warp 大小：GPU 以 Warp（通常 32 個執行緒）為單位排程執行
        printf("  Warp Size: %d\n", deviceProp.warpSize);

        printf("\n");
    }

    printf("========================================\n");
    printf("CUDA Environment Verification Complete!\n");
    printf("========================================\n");

    return 0;
}
