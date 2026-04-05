#include <stdio.h>

/**
 * 展示 Warp 資訊
 * 每個 Warp 包含 32 個執行緒，是 GPU 排程的最小單位
 */
// __global__ 表示這是一個 GPU 核心函式，由 CPU 端呼叫、在 GPU 上執行
__global__ void showWarpInfo() {
    // threadIdx.x 是執行緒在 Block 內的索引（從 0 開始）
    int warpId = threadIdx.x / 32;   // 每 32 個執行緒組成一個 Warp
    int laneId = threadIdx.x % 32;   // Lane ID：執行緒在 Warp 中的位置（0~31）

    // 只讓每個 Warp 的第一個執行緒（lane 0）輸出，避免重複列印
    if (laneId == 0) {
        // blockIdx.x 是此 Block 在 Grid 中的索引
        // blockDim.x 是每個 Block 的執行緒數量
        printf("Block %d, Warp %d (threads %d-%d)\n",
               blockIdx.x, warpId,
               blockIdx.x * blockDim.x + warpId * 32,
               blockIdx.x * blockDim.x + warpId * 32 + 31);
    }
}

/**
 * 展示所有執行緒的 Warp 資訊（詳細版本）
 * 每個執行緒都會印出自己的位置資訊
 */
__global__ void showDetailedWarpInfo() {
    // 計算全域索引：Block 編號 × 每個 Block 的大小 + Block 內的執行緒編號
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int warpId = threadIdx.x / 32;
    int laneId = threadIdx.x % 32;

    printf("Global ID: %3d | Block: %d | Thread: %3d | Warp: %d | Lane: %2d\n",
           idx, blockIdx.x, threadIdx.x, warpId, laneId);
}

int main() {
    printf("========================================\n");
    printf("    Warp 資訊示範\n");
    printf("========================================\n\n");

    // 示範 1：Block 大小 = 64（2 個 Warp）
    printf("示範 1: Block 大小 = 64\n");
    printf("配置: <<<2, 64>>>\n");
    printf("每個 Block 有 64/32 = 2 個 Warp\n");
    printf("----------------------------------------\n");
    // <<<2, 64>>> 表示啟動 2 個 Block，每個 Block 有 64 個執行緒
    showWarpInfo<<<2, 64>>>();
    // cudaDeviceSynchronize() 讓 CPU 等待 GPU 所有工作完成
    cudaDeviceSynchronize();

    // 示範 2：Block 大小 = 128（4 個 Warp）
    printf("\n示範 2: Block 大小 = 128\n");
    printf("配置: <<<1, 128>>>\n");
    printf("每個 Block 有 128/32 = 4 個 Warp\n");
    printf("----------------------------------------\n");
    showWarpInfo<<<1, 128>>>();
    cudaDeviceSynchronize();

    // 示範 3：Block 大小不是 32 的倍數
    printf("\n示範 3: Block 大小 = 100（不是 32 的倍數）\n");
    printf("配置: <<<1, 100>>>\n");
    printf("需要 ceil(100/32) = 4 個 Warp\n");
    printf("最後一個 Warp 只有 100-96 = 4 個活躍執行緒\n");
    printf("----------------------------------------\n");
    // ⚠️ 注意：Block 大小不是 32 的倍數時，最後一個 Warp 會有閒置的執行緒，浪費 GPU 資源
    showWarpInfo<<<1, 100>>>();
    cudaDeviceSynchronize();

    // 示範 4：詳細輸出（小規模）
    printf("\n示範 4: 詳細執行緒資訊（Block 大小 = 8）\n");
    printf("注意: Warp 大小仍是 32，但只有 8 個執行緒活躍\n");
    printf("----------------------------------------\n");
    showDetailedWarpInfo<<<1, 8>>>();
    cudaDeviceSynchronize();

    printf("\n========================================\n");
    printf("💡 重要概念：\n");
    printf("1. Warp 是 GPU 執行的基本單位（32 threads）\n");
    printf("2. Block 大小應該是 32 的倍數以提高效率\n");
    printf("3. Lane ID 是執行緒在 Warp 中的位置（0-31）\n");
    printf("========================================\n");

    return 0;
}
